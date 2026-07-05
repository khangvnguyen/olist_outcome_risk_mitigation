"""
Feature engineering stage: turns the shared order-level table (see
src/data/loader.py) into a model-ready, leakage-safe feature table.

Run:
    python -m src.features.build_features

Outputs (under output/features/):
    feature_table.csv   one row per order with a known review score
    schema.md            column-by-column description + eligibility notes

Leakage control (see docs/problem_framing.md section 2): seller/product/
category "history" features must only reflect orders that happened BEFORE
the order being scored. This is enforced two ways:
  1. Mechanically -- all history stats below are computed as prior-only
     cumulative sums over the order table sorted by order_purchase_timestamp
     (see `_prior_cumsum`), never a "leave-one-out on the full dataset" mean.
  2. By an explicit allowlist (ALLOWED_FEATURE_COLUMNS) -- the final table's
     columns are asserted to be a subset of it, so an accidental leaky join
     fails loudly instead of silently shipping.

Known limitation (documented, not silently ignored): the cutoff above is
based on order_purchase_timestamp, not on when an order's outcome actually
became KNOWN (review_answer_timestamp for bad_review, or
order_delivered_customer_date for is_late). Those lag purchase by days to
weeks, so a handful of "prior" orders used in a given row's history may not
have actually been resolved yet at that point in real time. A stricter
version would cut off on resolution time (would need a per-entity
merge-asof rather than a plain cumulative sum). Left as a next step --
see README "What's next".
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import RAW_DIR, load_raw, build_order_level_table

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "features"

K_SELLER = 20    # shrinkage constant, seller rate -> global rate
K_PRODUCT = 20   # shrinkage constant, product rate -> category rate
TEST_FRACTION = 0.2

METADATA_COLUMNS = ["order_id", "order_purchase_timestamp", "customer_unique_id", "split"]
TARGET_COLUMN = "bad_review"

ALLOWED_FEATURE_COLUMNS = [
    "primary_category",
    "total_item_price",
    "total_freight_value",
    "primary_payment_type",
    "max_installments",
    "customer_state",
    "customer_seller_distance_km",
    "estimated_delivery_days",
    "purchase_month",
    "purchase_season",
    "seller_bad_review_rate_smoothed",
    "seller_late_rate_smoothed",
    "seller_n_orders_to_date",
    "product_bad_review_rate_smoothed",
    "product_n_orders_to_date",
]

# Defense in depth: these must never appear in the output, even if someone
# extends ALLOWED_FEATURE_COLUMNS later without re-reading the leakage notes.
BANNED_COLUMNS = [
    "is_late", "days_late", "is_canceled_or_unavailable", "order_status",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_approved_at", "review_score", "review_comment_title",
    "review_comment_message", "has_review_comment",
]

SCHEMA = [
    ("order_id", "str", "Order identifier (metadata, not a feature)."),
    ("order_purchase_timestamp", "datetime", "Purchase time (metadata; also the basis for the point-in-time cutoff and the train/test split)."),
    ("customer_unique_id", "str", "Real-customer identifier across orders (metadata, kept for potential future churn analysis)."),
    ("split", "str", "'train'/'test', time-based: last 20% of orders by purchase date are test. Recommended for out-of-time evaluation; modeling stage may override."),
    ("bad_review", "bool", "TARGET. review_score <= 2. Rows with no review score are dropped before this table is built."),
    ("primary_category", "category", "English product category of the order's highest-price item. Missing categories filled as 'unknown' (a real, if imprecise, bucket -- ~1.85% of products)."),
    ("total_item_price", "float", "Sum of item prices across the order's line items. Known at order time."),
    ("total_freight_value", "float", "Sum of freight across the order's line items. Known at order time."),
    ("primary_payment_type", "category", "Most-used payment method for the order."),
    ("max_installments", "int", "Max installments chosen across the order's payment rows."),
    ("customer_state", "category", "Customer's Brazilian state."),
    ("customer_seller_distance_km", "float", "Haversine distance, customer zip centroid to (primary) seller zip centroid. NaN where either zip has no geolocation match (~few % of orders) -- left as NaN, not imputed here."),
    ("estimated_delivery_days", "float", "order_estimated_delivery_date - order_purchase_timestamp, in days. Uses the ESTIMATE, known at order time -- not the actual delivered date."),
    ("purchase_month", "int", "Calendar month (1-12) of order_purchase_timestamp."),
    ("purchase_season", "category", "Southern-hemisphere (Brazil) season derived from purchase_month: summer/autumn/winter/spring."),
    ("seller_bad_review_rate_smoothed", "float", f"Seller's point-in-time bad_review rate over all its PRIOR orders, shrunk toward the point-in-time global rate with k={K_SELLER} (formula: (bad_sum_prior + k*global_rate_prior) / (n_prior + k)). Cold-start (no prior orders) reduces cleanly to the global rate."),
    ("seller_late_rate_smoothed", "float", f"Same construction as above but for is_late, over the seller's prior DELIVERED orders only, shrunk toward the point-in-time global late rate with k={K_SELLER}. Diagnostic proxy for delivery reliability -- is_late itself is never used as a feature."),
    ("seller_n_orders_to_date", "int", "Count of the seller's prior orders with a known bad_review outcome, as of this order's purchase time. Also usable as a raw 'how much history do we have' signal."),
    ("product_bad_review_rate_smoothed", "float", f"Product's point-in-time bad_review rate over its PRIOR orders, shrunk toward the point-in-time CATEGORY rate with k={K_PRODUCT} (95% of products have <10 prior orders, so the raw per-product rate is mostly noise -- see docs/eda_findings.md)."),
    ("product_n_orders_to_date", "int", "Count of the product's prior orders with a known bad_review outcome, as of this order's purchase time."),
]

SEASON_MAP = {
    12: "summer", 1: "summer", 2: "summer",
    3: "autumn", 4: "autumn", 5: "autumn",
    6: "winter", 7: "winter", 8: "winter",
    9: "spring", 10: "spring", 11: "spring",
}


def _shrink_toward_baseline(cum_prior, n_prior, baseline_rate, k):
    """Shrinkage-toward-baseline estimate: blend an entity's own prior-orders
    rate with a baseline rate (global or category), weighted by how much
    prior history the entity has (`k` = the "worth this many baseline
    orders" constant). Computed from sums directly rather than
    `n_prior * raw_rate` so that `n_prior == 0` (no history yet) can't
    silently multiply a NaN raw rate into the result -- it correctly
    reduces to `baseline_rate` instead."""
    return (cum_prior + k * baseline_rate) / (n_prior + k)


def _prior_cumsum(df: pd.DataFrame, group_col: str, value_col: str, valid_col: str):
    """Prior-only (excludes current row) cumulative sum/count of `value_col`,
    within groups of `group_col`, given df already sorted by purchase time.
    Computed as cumsum-including-self minus self, which sidesteps
    groupby+shift edge cases at group boundaries. Pass group_col=None for a
    dataset-wide (ungrouped) cumulative sum."""
    if group_col is None:
        cum_val_incl = df[value_col].cumsum()
        cum_n_incl = df[valid_col].cumsum()
    else:
        grp = df.groupby(group_col)
        cum_val_incl = grp[value_col].cumsum()
        cum_n_incl = grp[valid_col].cumsum()
    val_prior = cum_val_incl - df[value_col]
    n_prior = cum_n_incl - df[valid_col]
    return val_prior, n_prior


def build_feature_table(raw: dict) -> pd.DataFrame:
    df = build_order_level_table(raw)

    n_before = len(df)
    df = df[df["bad_review"].notna()].copy()
    print(f"[features] Dropped {n_before - len(df):,} orders with no review score "
          f"({len(df):,} remain).")

    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

    df["primary_category"] = df["primary_category"].fillna("unknown")

    # Each outcome is split into a 0/1 "_val" (NaN treated as 0, only safe
    # because "_valid" tracks which rows actually count) and a "_valid" mask
    # (1 = outcome is known). _prior_cumsum() below sums both in lockstep, so
    # an order with an unresolved outcome (e.g. no review yet) contributes 0
    # to the numerator AND 0 to the denominator -- "not yet observed", not
    # "observed as good".
    df["_bad_val"] = np.where(df["bad_review"].notna(), df["bad_review"].astype(float), 0.0)
    df["_bad_valid"] = df["bad_review"].notna().astype(int)
    df["_late_val"] = np.where(df["is_late"].notna(), df["is_late"].astype(float), 0.0)
    df["_late_valid"] = df["is_late"].notna().astype(int)

    # global (dataset-wide) prior rates -- used as shrinkage baselines
    df["global_bad_cum_prior"], df["global_n_cum_prior"] = _prior_cumsum(df, None, "_bad_val", "_bad_valid")
    df["global_rate_prior"] = df["global_bad_cum_prior"] / df["global_n_cum_prior"].replace(0, np.nan)

    df["global_late_cum_prior"], df["global_late_n_cum_prior"] = _prior_cumsum(df, None, "_late_val", "_late_valid")
    df["global_late_rate_prior"] = df["global_late_cum_prior"] / df["global_late_n_cum_prior"].replace(0, np.nan)

    # category prior rate -- shrinkage baseline for product-level rate
    df["cat_bad_cum_prior"], df["cat_n_cum_prior"] = _prior_cumsum(df, "primary_category", "_bad_val", "_bad_valid")
    df["cat_rate_prior"] = df["cat_bad_cum_prior"] / df["cat_n_cum_prior"].replace(0, np.nan)

    # seller prior stats (bad_review and late, separately)
    df["seller_bad_cum_prior"], df["seller_n_cum_prior"] = _prior_cumsum(df, "primary_seller_id", "_bad_val", "_bad_valid")
    df["seller_late_cum_prior"], df["seller_late_n_cum_prior"] = _prior_cumsum(df, "primary_seller_id", "_late_val", "_late_valid")

    # product prior stats
    df["product_bad_cum_prior"], df["product_n_cum_prior"] = _prior_cumsum(df, "primary_product_id", "_bad_val", "_bad_valid")

    # sanity check: prior counts must be monotonically non-decreasing within
    # each entity's own timeline (they're cumulative by construction -- a
    # violation would mean the sort-then-groupby machinery above is broken)
    for group_col, col in [("primary_seller_id", "seller_n_cum_prior"), ("primary_product_id", "product_n_cum_prior")]:
        diffs = df.groupby(group_col)[col].diff()
        assert (diffs.dropna() >= 0).all(), f"{col} is not monotonic within {group_col} -- point-in-time logic is broken"
    print("[features] Sanity check passed: prior-count columns are monotonic non-decreasing per entity.")

    # time-based train/test split, computed before using train-only fallback means
    cutoff = df["order_purchase_timestamp"].quantile(1 - TEST_FRACTION)
    df["split"] = np.where(df["order_purchase_timestamp"] >= cutoff, "test", "train")
    print(f"[features] Time-based split at {cutoff.date()}: "
          f"{(df['split'] == 'train').sum():,} train / {(df['split'] == 'test').sum():,} test.")

    # fallback constants for the handful of rows at the very start of the
    # dataset's timeline where even the global prior rate is undefined (n=0)
    train_bad_mean = df.loc[df["split"] == "train", "bad_review"].astype(float).mean()
    train_late_mean = df.loc[df["split"] == "train", "is_late"].astype(float).mean()

    global_rate_filled = df["global_rate_prior"].fillna(train_bad_mean)
    global_late_rate_filled = df["global_late_rate_prior"].fillna(train_late_mean)
    cat_rate_filled = df["cat_rate_prior"].fillna(global_rate_filled)

    df["seller_bad_review_rate_smoothed"] = _shrink_toward_baseline(
        df["seller_bad_cum_prior"], df["seller_n_cum_prior"], global_rate_filled, K_SELLER
    )
    df["seller_late_rate_smoothed"] = _shrink_toward_baseline(
        df["seller_late_cum_prior"], df["seller_late_n_cum_prior"], global_late_rate_filled, K_SELLER
    )
    df["product_bad_review_rate_smoothed"] = _shrink_toward_baseline(
        df["product_bad_cum_prior"], df["product_n_cum_prior"], cat_rate_filled, K_PRODUCT
    )
    df["seller_n_orders_to_date"] = df["seller_n_cum_prior"]
    df["product_n_orders_to_date"] = df["product_n_cum_prior"]

    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_season"] = df["purchase_month"].map(SEASON_MAP)

    final_cols = METADATA_COLUMNS + [TARGET_COLUMN] + ALLOWED_FEATURE_COLUMNS
    out = df[final_cols].copy()

    banned_present = [c for c in BANNED_COLUMNS if c in out.columns]
    assert not banned_present, f"Leaked post-outcome columns present in feature table: {banned_present}"
    assert set(out.columns) <= set(METADATA_COLUMNS + [TARGET_COLUMN] + ALLOWED_FEATURE_COLUMNS), \
        "Feature table has columns outside the explicit allowlist -- check ALLOWED_FEATURE_COLUMNS."

    return out


def write_schema(path: Path):
    lines = [
        "# Feature Table Schema\n",
        "Auto-generated by `src/features/build_features.py`. Do not hand-edit.\n",
        "One row per order with a known review score. Columns below are either "
        "metadata, the target, or pre-outcome-eligible features -- see "
        "`docs/problem_framing.md` section 2 for the eligibility rule and "
        "`src/features/build_features.py`'s module docstring for the "
        "point-in-time leakage-control approach and its known limitations.\n",
        "| Column | Type | Description |",
        "|---|---|---|",
    ]
    for name, dtype, desc in SCHEMA:
        lines.append(f"| `{name}` | {dtype} | {desc} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    print("[features] Loading raw tables...")
    raw = load_raw(RAW_DIR)

    print("[features] Building feature table...")
    feature_table = build_feature_table(raw)
    print(f"[features] Feature table: {feature_table.shape[0]:,} rows x {feature_table.shape[1]} cols")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "feature_table.csv"
    feature_table.to_csv(out_path, index=False)
    write_schema(OUTPUT_DIR / "schema.md")

    print(f"[features] Done. See {out_path} and {OUTPUT_DIR / 'schema.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
