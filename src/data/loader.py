"""
Shared loading / joining logic for the Olist dataset.

Kept separate from both eda/ and features/ so both stages build their
tables the same way -- avoids subtle inconsistencies between "what EDA
looked at" and "what the model was trained on".

IMPORTANT (leakage note): build_order_level_table() below includes both
pre-outcome and post-outcome columns (e.g. actual delivery dates,
review_score). This is intentional -- EDA needs to see outcomes to
understand them. The feature-eligibility split (which columns a MODEL is
allowed to use) is enforced separately in src/features/, not here.
See docs/problem_framing.md section 2.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

CANCELED_STATUSES = {"canceled", "unavailable"}


def load_raw(raw_dir: Path = RAW_DIR) -> dict:
    """Load all 9 CSVs into a dict of DataFrames, with light dtype control
    on the largest file (geolocation) to keep memory reasonable."""
    dtypes_geo = {
        "geolocation_zip_code_prefix": "int32",
        "geolocation_lat": "float32",
        "geolocation_lng": "float32",
        "geolocation_city": "category",
        "geolocation_state": "category",
    }

    raw = {
        "orders": pd.read_csv(raw_dir / "olist_orders_dataset.csv"),
        "customers": pd.read_csv(raw_dir / "olist_customers_dataset.csv"),
        "order_items": pd.read_csv(raw_dir / "olist_order_items_dataset.csv"),
        "order_payments": pd.read_csv(raw_dir / "olist_order_payments_dataset.csv"),
        "order_reviews": pd.read_csv(raw_dir / "olist_order_reviews_dataset.csv"),
        "products": pd.read_csv(raw_dir / "olist_products_dataset.csv"),
        "sellers": pd.read_csv(raw_dir / "olist_sellers_dataset.csv"),
        "category_translation": pd.read_csv(raw_dir / "product_category_name_translation.csv"),
        "geolocation": pd.read_csv(raw_dir / "olist_geolocation_dataset.csv", dtype=dtypes_geo),
    }

    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for c in date_cols:
        raw["orders"][c] = pd.to_datetime(raw["orders"][c], errors="coerce")

    raw["order_reviews"]["review_creation_date"] = pd.to_datetime(
        raw["order_reviews"]["review_creation_date"], errors="coerce"
    )
    raw["order_reviews"]["review_answer_timestamp"] = pd.to_datetime(
        raw["order_reviews"]["review_answer_timestamp"], errors="coerce"
    )
    raw["order_items"]["shipping_limit_date"] = pd.to_datetime(
        raw["order_items"]["shipping_limit_date"], errors="coerce"
    )

    return raw


def build_zip_geo_lookup(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Collapse the geolocation table to one row per zip_code_prefix using
    the centroid (mean lat/lng) of all points sharing that prefix, after
    dropping exact duplicate rows. This is necessary because the raw table
    has ~26% duplicate rows and multiple lat/lng pairs per prefix."""
    geo = geolocation.drop_duplicates()
    lookup = (
        geo.groupby("geolocation_zip_code_prefix", observed=True)
        .agg(
            lat=("geolocation_lat", "mean"),
            lng=("geolocation_lng", "mean"),
            city=("geolocation_city", lambda s: s.mode().iat[0] if not s.mode().empty else np.nan),
            state=("geolocation_state", lambda s: s.mode().iat[0] if not s.mode().empty else np.nan),
        )
        .reset_index()
        .rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"})
    )
    return lookup


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Vectorised haversine distance in km. Any NaN input -> NaN output."""
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def build_order_items_agg(order_items: pd.DataFrame, products: pd.DataFrame,
                            category_translation: pd.DataFrame) -> pd.DataFrame:
    """One row per order_id, aggregating the (possibly multi-row) order_items
    table: total price/freight, item/seller/product counts, and a
    "primary" product category (by total spend within the order)."""
    items = order_items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    items = items.merge(category_translation, on="product_category_name", how="left")

    # primary category = category of the line with the highest price in the order
    idx = items.groupby("order_id")["price"].idxmax()
    primary_cat = items.loc[idx, ["order_id", "product_category_name_english"]].rename(
        columns={"product_category_name_english": "primary_category"}
    )

    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_distinct_products=("product_id", "nunique"),
        n_distinct_sellers=("seller_id", "nunique"),
        total_item_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
    ).reset_index()

    agg = agg.merge(primary_cat, on="order_id", how="left")
    return agg


def build_payments_agg(order_payments: pd.DataFrame) -> pd.DataFrame:
    """One row per order_id: total paid, max installments chosen, and the
    most-used payment type (by count of payment rows, not value)."""
    agg = order_payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        n_payment_methods=("payment_type", "nunique"),
    ).reset_index()

    mode_type = (
        order_payments.groupby("order_id")["payment_type"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
        .reset_index()
        .rename(columns={"payment_type": "primary_payment_type"})
    )
    return agg.merge(mode_type, on="order_id", how="left")


def build_order_level_table(raw: dict) -> pd.DataFrame:
    """Assemble a single order-level DataFrame joining all 9 tables.

    Includes BOTH pre-outcome and post-outcome columns -- see module
    docstring. Target candidates are computed here for convenience:
      - bad_review: review_score <= 2 (NaN if no review)
      - is_late: delivered after estimated date (NaN if not delivered)
      - is_canceled_or_unavailable: order_status in {canceled, unavailable}
    """
    orders = raw["orders"]
    customers = raw["customers"]
    reviews = raw["order_reviews"].drop_duplicates(subset="order_id", keep="last")
    items_agg = build_order_items_agg(raw["order_items"], raw["products"], raw["category_translation"])
    payments_agg = build_payments_agg(raw["order_payments"])
    geo_lookup = build_zip_geo_lookup(raw["geolocation"])

    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(
        reviews[["order_id", "review_score", "review_comment_title", "review_comment_message"]],
        on="order_id", how="left",
    )
    df = df.merge(items_agg, on="order_id", how="left")
    df = df.merge(payments_agg, on="order_id", how="left")

    # customer geolocation (by zip prefix)
    df = df.merge(
        geo_lookup.rename(columns={"zip_code_prefix": "customer_zip_code_prefix",
                                     "lat": "customer_lat", "lng": "customer_lng"})[
            ["customer_zip_code_prefix", "customer_lat", "customer_lng"]
        ],
        on="customer_zip_code_prefix", how="left",
    )

    # target candidates
    # Note: using pandas' nullable "boolean" dtype (not plain numpy bool/float)
    # so that missing values coexist with True/False *and* groupby/comparisons
    # against literal True/False work correctly. A plain np.where(...) here
    # would silently produce float64 (0.0/1.0/nan), where e.g. series.get(False)
    # fails to match the 0.0 index label -- a real bug caught during testing.
    df["bad_review"] = pd.array(
        np.where(df["review_score"].notna(), df["review_score"] <= 2, pd.NA),
        dtype="boolean",
    )
    df["is_late"] = pd.array(
        np.where(
            df["order_delivered_customer_date"].notna(),
            df["order_delivered_customer_date"] > df["order_estimated_delivery_date"],
            pd.NA,
        ),
        dtype="boolean",
    )
    df["days_late"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.total_seconds() / 86400
    df["is_canceled_or_unavailable"] = df["order_status"].isin(CANCELED_STATUSES)
    df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    df["estimated_delivery_days"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    df["has_review_comment"] = df["review_comment_message"].notna()

    return df
