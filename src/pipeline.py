"""
Single entrypoint for the whole solution.

    python -m src.pipeline

Runs, in order:
    1. Data acquisition (download if missing, else verify local files)
    2. [TODO] EDA -> output/eda/
    3. [TODO] Modeling & evaluation -> output/model/
    4. [TODO] Final report -> output/report.md

This is intentionally a thin orchestrator. Each stage lives in its own
module under src/ and can be run/tested independently; this file just
sequences them and is what `docker compose up` calls.
"""

import sys

from src.data.download import main as download_main
from src.eda.run_eda import main as eda_main


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

    print("\n[stage 3/4] Modeling -- not yet implemented")
    print("[stage 4/4] Report -- not yet implemented")

    print("\n[pipeline] Done (partial -- data + EDA stages only so far).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
