"""
Feature engineering stage: turns the shared order-level table (see
src/data/loader.py) into a model-ready, leakage-safe feature table.

Run:
    python -m src.features.build_features

Outputs (under output/features/):
    feature_table.csv    one row per order with a known review score
    schema.md            column-by-column description + eligibility notes

WHY LEAKAGE IS THE MAIN RISK HERE
----------------------------------
Several features are "history" stats -- e.g. "this seller's bad-review rate
so far". If we compute that rate using ALL of the seller's orders (past
*and* future relative to the order we're scoring), we'd be leaking
information from the future into a feature. So every history feature below
is computed using only orders that happened BEFORE the order being scored
("prior" orders).

We enforce that in two ways:
  1. Mechanically: history stats are running totals computed row-by-row
     over the table sorted by order_purchase_timestamp, where each row only
     sees orders strictly before it (see `_prior_cumsum`). We never compute
     a "leave-one-out mean over the whole dataset", which would be an easy
     way to accidentally leak the future in.
  2. With an allowlist: ALLOWED_FEATURE_COLUMNS lists every column that's
     allowed in the final table. Before saving, we assert the output only
     contains columns from that list, so if someone later joins in a leaky
     column by accident, the script crashes instead of silently shipping it.

KNOWN LIMITATION (documented on purpose, not an oversight)
------------------------------------------------------------
The "prior order" cutoff uses order_purchase_timestamp -- i.e. we treat an
order's outcome as known as soon as it was purchased. In reality, the
outcome (e.g. whether the review was bad) isn't observed until later:
review_answer_timestamp for bad_review, or order_delivered_customer_date
for is_late. That gap is days to weeks. So a few "prior" orders used in a
given row's history stats may not actually have had a resolved outcome yet
at that real point in time -- a small amount of look-ahead leakage.

Fixing this properly would mean cutting off on *resolution* time instead of
purchase time, which requires an as-of merge per entity rather than a plain
running total. Left as a next step -- see README "What's next".
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import RAW_DIR, load_raw, build_order_level_table

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "features"

K_SELLER = 20    # shrinkage constant: how many "baseline-strength" orders
                 # a seller's own history is worth before we trust it fully
K_PRODUCT = 20   # same idea, but for a product shrinking toward its category
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
# extends ALLOWED_FEATURE_COLUMNS later without re-reading the leakage notes
# above. If any of these sneak in, the script should crash, not ship.
BANNED_COLUMNS = [
    "is_late", "days_late", "is_canceled_or_unavailable", "order_status",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_approved_at", "review_score", "review_comment_title",
    "review_comment_message", "has_review_comment",
]

SCHEMA = [
    ("order_id", "str", "Order identifier (metadata, not a feature)."),
    ("order_purchase_timestamp", "datetime", "When the order was placed. Used as metadata, and also as the cutoff for both the point-in-time history features and the train/test split."),
    ("customer_unique_id", "str", "Customer identifier, consistent across their orders (metadata; kept in case we want to look at repeat-customer behavior later)."),
    ("split", "str", "'train' or 'test'. Time-based: the most recent 20% of orders (by purchase date) are 'test', the rest 'train'. This gives an out-of-time evaluation, which is more realistic than a random split for a model that will predict on future orders. The modeling stage is free to re-split if needed."),
    ("bad_review", "bool", "TARGET. True if review_score <= 2. Orders with no review score are dropped before this table is built, since we can't label them."),
    ("primary_category", "category", "Product category of the order's highest-priced item (in English). Missing categories are filled as 'unknown' rather than dropped -- about 1.85% of products have no category, and that's arguably useful signal in itself."),
    ("total_item_price", "float", "Sum of item prices across the order's line items. Known at the time the order is placed."),
    ("total_freight_value", "float", "Sum of freight/shipping charges across the order's line items. Known at order time."),
    ("primary_payment_type", "category", "The payment method used most within the order (an order can have multiple payment rows)."),
    ("max_installments", "int", "The largest number of installments chosen across the order's payment rows."),
    ("customer_state", "category", "Brazilian state of the customer."),
    ("customer_seller_distance_km", "float", "Straight-line (haversine) distance between the customer's and seller's zip-code centroids. Left as NaN (not imputed) when either zip has no geolocation match, which happens for a small percentage of orders."),
    ("estimated_delivery_days", "float", "Number of days between order_estimated_delivery_date and order_purchase_timestamp. This is the ESTIMATE given at order time, not how long delivery actually took -- the actual delivery date is an outcome and is never used here."),
    ("purchase_month", "int", "Calendar month (1-12) the order was placed."),
    ("purchase_season", "category", "Season in Brazil (southern hemisphere) corresponding to purchase_month: summer, autumn, winter, or spring."),
    ("seller_bad_review_rate_smoothed", "float", f"How often this seller's PAST orders (before this one) resulted in a bad review, adjusted so sellers with little history aren't over- or under-trusted. Concretely: (seller's past bad-review count + {K_SELLER} x current global bad-review rate) / (seller's past order count + {K_SELLER}). A seller with zero prior orders gets exactly the global rate; a seller with lots of history gets a value close to their own true rate. This is standard 'shrinkage' / 'smoothing' toward a baseline."),
    ("seller_late_rate_smoothed", "float", f"Same idea as seller_bad_review_rate_smoothed, but for late deliveries (is_late) instead of bad reviews, using only the seller's past DELIVERED orders. This exists as a diagnostic signal about seller reliability -- is_late itself is an outcome and is never used directly as a feature."),
    ("seller_n_orders_to_date", "int", "How many of the seller's past orders already have a known bad_review outcome, as of this order. Doubles as a 'how much do we actually know about this seller' signal -- useful on its own, and it's also the denominator behind seller_bad_review_rate_smoothed."),
    ("product_bad_review_rate_smoothed", "float", f"Same shrinkage idea as the seller version, but for this specific product, shrunk toward its CATEGORY's rate (not the global rate) since categories are a closer baseline for a product's typical rate. Formula: (product's past bad-review count + {K_PRODUCT} x current category bad-review rate) / (product's past order count + {K_PRODUCT}). This matters a lot here because ~95% of products have fewer than 10 prior orders, so a product's raw own-rate would mostly be noise -- see docs/eda_findings.md."),
    ("product_n_orders_to_date", "int", "How many of this product's past orders already have a known bad_review outcome, as of this order."),
]

SEASON_MAP = {
    12: "summer", 1: "summer", 2: "summer",
    3: "autumn", 4: "autumn", 5: "autumn",
    6: "winter", 7: "winter", 8: "winter",
    9: "spring", 10: "spring", 11: "spring",
}


def _shrink_toward_baseline(cum_prior, n_prior, baseline_rate, k):
    """
    If a seller/product has very little history, do not fully trust its raw rate.
    Blend its own history with a safer baseline.

    Formula: (cum_prior + k * baseline_rate) / (n_prior + k)

    For seller bad-review rate:
    - cum_prior = number of prior bad reviews for this seller
    - n_prior = number of prior known reviewed orders for this seller
    - baseline_rate = global bad-review rate before this order
    - k = 20 = treat the baseline as if it contributes 20 “imaginary” prior orders

    Example:
    A seller has only 2 prior orders, 1 bad review.

    Raw seller rate:
    1 / 2 = 50%
    That is probably too extreme because 2 orders is tiny.
    If the global prior bad-review rate is 15%, with k=20:
    (1 + 20 * 0.15) / (2 + 20)
    = (1 + 3) / 22
    = 18.2%
    So the seller is treated as somewhat risky, but not 50% risky.

    If the seller has 200 prior orders and 60 bad reviews:
    (60 + 20 * 0.15) / (200 + 20)
    = 63 / 220
    = 28.6%
    That is close to the seller’s raw 30%, because with lots of history we trust the seller’s own data more.

    This is called shrinkage because small-sample estimates are pulled back toward a baseline.
    """
    return (cum_prior + k * baseline_rate) / (n_prior + k)


def _prior_cumsum(df: pd.DataFrame, group_col: str, value_col: str, valid_col: str):
    """
    For each row, count/sum only the previous rows, not the current row,
    optionally within a group like seller or product

    Example, for one seller:
    order time	seller	bad_review
    Jan 1	    A	    0
    Jan 5	    A	    1
    Jan 9	    A	    0
    Jan 12	    A	    1

    For the Jan 12 order, prior history should be:
    prior bad reviews = 1
    prior known orders = 3
    prior bad-review rate = 1 / 3 = 33%
    It should not include the Jan 12 order itself, because that would leak the target.

    Implementation note: we compute the INCLUDING-current-row cumulative
    sum first (pandas' built-in .cumsum(), which is fast and well-tested),
    then subtract the current row's own value to exclude it.
    (meaning it first calculates cumulative history including the current row,
    then subtracts the current row to get prior-only history.)

    Returns a (value_prior, count_prior) pair of Series, aligned to df's
    index.
    """
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

    # For each outcome we track (bad_review, is_late), build two helper
    # columns: a "_val" column (the 0/1 outcome, with NaN replaced by 0) and
    # a "_valid" column (1 if the outcome is actually known, else 0).

    # Example: 
    # Suppose a seller has these three prior orders:
    # order	   bad_review
    # 1	       False
    # 2	       True
    # 3	       missing

    # We do not want the missing to be counted.
    # If we don't have this part,
    # we will calculate bad review rate = (1 bad) / (3 rows) = 33%
    # But if we implement this part of code,
    # bad review rate = sum(_bad_val) / sum(_bad_valid) = (0 + 1 + 0) / (1 + 1 + 0) = 1/2 = 50%
    # which is the correct calculation

    df["_bad_val"] = np.where(df["bad_review"].notna(), df["bad_review"].astype(float), 0.0)
    df["_bad_valid"] = df["bad_review"].notna().astype(int)
    df["_late_val"] = np.where(df["is_late"].notna(), df["is_late"].astype(float), 0.0)
    df["_late_valid"] = df["is_late"].notna().astype(int)

    # Global (whole-dataset) prior rate -- the baseline that seller-level
    # rates shrink toward.
    df["global_bad_cum_prior"], df["global_n_cum_prior"] = _prior_cumsum(df, None, "_bad_val", "_bad_valid")
    df["global_rate_prior"] = df["global_bad_cum_prior"] / df["global_n_cum_prior"].replace(0, np.nan)

    df["global_late_cum_prior"], df["global_late_n_cum_prior"] = _prior_cumsum(df, None, "_late_val", "_late_valid")
    df["global_late_rate_prior"] = df["global_late_cum_prior"] / df["global_late_n_cum_prior"].replace(0, np.nan)

    # Category-level prior rate -- the baseline that product-level rates
    # shrink toward (a product's category is a much better baseline than
    # the dataset-wide rate).
    df["cat_bad_cum_prior"], df["cat_n_cum_prior"] = _prior_cumsum(df, "primary_category", "_bad_val", "_bad_valid")
    df["cat_rate_prior"] = df["cat_bad_cum_prior"] / df["cat_n_cum_prior"].replace(0, np.nan)

    # Seller-level prior stats, for both outcomes.
    df["seller_bad_cum_prior"], df["seller_n_cum_prior"] = _prior_cumsum(df, "primary_seller_id", "_bad_val", "_bad_valid")
    df["seller_late_cum_prior"], df["seller_late_n_cum_prior"] = _prior_cumsum(df, "primary_seller_id", "_late_val", "_late_valid")

    # Product-level prior stats.
    df["product_bad_cum_prior"], df["product_n_cum_prior"] = _prior_cumsum(df, "primary_product_id", "_bad_val", "_bad_valid")

    # Sanity check: since these are running totals over each entity's own
    # timeline, they can only stay flat or increase row-to-row -- never
    # decrease. If they ever decrease, the sort-then-groupby logic above is
    # broken.
    for group_col, col in [("primary_seller_id", "seller_n_cum_prior"), ("primary_product_id", "product_n_cum_prior")]:
        diffs = df.groupby(group_col)[col].diff()
        assert (diffs.dropna() >= 0).all(), f"{col} is not monotonic within {group_col} -- point-in-time logic is broken"
    print("[features] Sanity check passed: prior-count columns are monotonic non-decreasing per entity.")

    # Time-based train/test split. Computed here, before we fall back to a
    # train-only mean below, so that fallback mean is never contaminated
    # by test-set data.
    cutoff = df["order_purchase_timestamp"].quantile(1 - TEST_FRACTION)
    df["split"] = np.where(df["order_purchase_timestamp"] >= cutoff, "test", "train")
    print(f"[features] Time-based split at {cutoff.date()}: "
          f"{(df['split'] == 'train').sum():,} train / {(df['split'] == 'test').sum():,} test.")

    # For the handful of rows right at the start of the dataset's timeline,
    # even the global/category prior rate is undefined (n=0, so 0/0). Fall
    # back to a fixed constant: the overall train-set mean.
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