"""
Single entrypoint for the whole solution.

    python -m src.pipeline

Runs, in order:
    1. Data acquisition (download if missing, else verify local files)
    2. EDA -> output/eda/
    3. Feature engineering -> output/features/
    4. Modeling & evaluation -> output/model/

This is intentionally a thin orchestrator. Each stage lives in its own
module under src/ and can be run/tested independently; this file just
sequences them and is what `docker compose up` calls.
"""

import sys

from src.data.download import main as download_main
from src.eda.run_eda import main as eda_main
from src.features.build_features import main as features_main
from src.models.train_model import main as model_main


def main():
    print("=" * 60)
    print("Olist Outcome Risk Pipeline")
    print("=" * 60)

    print("\n[stage 1/4] Data acquisition")
    rc = download_main()
    if rc != 0:
        print("[pipeline] Aborting: data acquisition failed. See message above.")
        return rc

    print("\n[stage 2/4] EDA")
    rc = eda_main()
    if rc != 0:
        print("[pipeline] Aborting: EDA failed. See message above.")
        return rc

    print("\n[stage 3/4] Feature engineering")
    rc = features_main()
    if rc != 0:
        print("[pipeline] Aborting: feature engineering failed. See message above.")
        return rc

    print("\n[stage 4/4] Modeling")
    rc = model_main()
    if rc != 0:
        print("[pipeline] Aborting: modeling failed. See message above.")
        return rc

    print("\n[pipeline] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
