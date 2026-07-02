"""
Download (or verify presence of) the Olist Brazilian E-Commerce dataset.

Two supported paths, tried in this order:

1. Kaggle API (automatic). Requires a Kaggle account + API token.
   Set up once:
     - Create a token at https://www.kaggle.com/settings -> "Create New Token"
       This downloads kaggle.json.
     - Place it at ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME / KAGGLE_KEY
       env vars) with permissions 600.
   Then this script will `kaggle datasets download` the dataset and unzip it
   into data/raw/.

2. Manual placement (fallback, no credentials needed). If Kaggle auth isn't
   configured, download the dataset yourself from:
     https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
   and place the 9 CSVs directly into data/raw/. This script will detect
   that they're already there and skip the download.

Either way, running `python -m src.data.download` (or the Docker entrypoint)
will end with either a populated data/raw/ or a clear error message telling
you what's missing.
"""

import os
import sys
import zipfile
from pathlib import Path

KAGGLE_DATASET = "olistbr/brazilian-ecommerce"

EXPECTED_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def files_present(raw_dir: Path) -> list:
    return [f for f in EXPECTED_FILES if not (raw_dir / f).exists()]


def try_kaggle_download(raw_dir: Path) -> bool:
    """Attempt to download via Kaggle API. Returns True on success."""
    try:
        # Imported here so the whole script doesn't hard-fail if the
        # kaggle package or credentials aren't available.
        # Note: kaggle's own __init__.py calls api.authenticate() at
        # import time and does `exit(1)` (raises SystemExit, not a normal
        # Exception) if no credentials are configured -- so SystemExit must
        # be caught here too, or a missing-credentials case would crash the
        # whole pipeline instead of falling through to manual instructions.
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[download] kaggle package not installed, skipping API download.")
        return False
    except (OSError, SystemExit) as e:
        # Raised by the kaggle package at import time if no credentials found.
        print(f"[download] Kaggle credentials not found ({e}), skipping API download.")
        return False

    # Note: the kaggle package raises SystemExit (not a regular Exception)
    # on auth failure, so it must be caught explicitly here too, or it will
    # kill the whole pipeline instead of falling back to manual instructions.
    try:
        api = KaggleApi()
        api.authenticate()
    except (Exception, SystemExit) as e:
        print(f"[download] Kaggle authentication failed: {e}")
        return False

    print(f"[download] Downloading {KAGGLE_DATASET} via Kaggle API...")
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        api.dataset_download_files(KAGGLE_DATASET, path=str(raw_dir), unzip=True)
    except (Exception, SystemExit) as e:
        print(f"[download] Kaggle download failed: {e}")
        return False

    return len(files_present(raw_dir)) == 0


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    missing = files_present(RAW_DIR)

    if not missing:
        print(f"[download] All {len(EXPECTED_FILES)} expected files already present in {RAW_DIR}. Skipping.")
        return 0

    print(f"[download] {len(missing)}/{len(EXPECTED_FILES)} files missing from {RAW_DIR}.")
    print("[download] Attempting automatic download via Kaggle API...")

    if try_kaggle_download(RAW_DIR):
        print("[download] Success. All files present.")
        return 0

    still_missing = files_present(RAW_DIR)
    if not still_missing:
        print("[download] Success. All files present.")
        return 0

    print(
        "\n[download] Could not obtain the dataset automatically.\n"
        "To proceed manually:\n"
        f"  1. Download from https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
        f"  2. Place these files directly in: {RAW_DIR}\n"
        f"     Missing: {', '.join(still_missing)}\n"
        "  3. Re-run this command (or the pipeline entrypoint).\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
