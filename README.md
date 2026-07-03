# Olist Outcome Risk — Predicting Poor Customer Experiences

> Status: data acquisition, EDA, feature engineering, and modeling stages
> are done. Only the final report/write-up stage remains (tracked in
> `docs/problem_framing.md` / `docs/eda_findings.md` / `docs/modeling_findings.md`).

## Setup

**Option A — automatic download (Kaggle API)**
1. Create a Kaggle API token: https://www.kaggle.com/settings → "Create New Token" → downloads `kaggle.json`
2. Place it at `~/.kaggle/kaggle.json`
3. Run `docker compose up --build`

**Option B — manual download**
1. Download the dataset from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Unzip the 9 CSVs into `data/raw/`
3. Run `docker compose up --build`

Either way, results land in `output/`.

## Repo structure
```
src/
  data/
    download.py       # data acquisition (Kaggle API, falls back to manual)
    loader.py          # shared raw-loading/joining logic (EDA + features build on this)
  eda/run_eda.py        # EDA -> output/eda/summary.md, tables/, figures/
  features/build_features.py  # leakage-safe feature table -> output/features/
  models/train_model.py # HGB risk model + seller risk table -> output/model/
  pipeline.py          # single entrypoint, orchestrates all stages
docs/
  problem_framing.md   # target definition + feature-leakage design doc
  eda_findings.md       # decision-ready EDA summary feeding the feature plan
  modeling_findings.md  # decision-ready modeling summary feeding the report
output/                # all results land here (eda/, features/, model/)
```

## Problem framing, key findings, design decisions

See `docs/problem_framing.md` (target definition, feature eligibility),
`docs/eda_findings.md` (what actually predicts `bad_review`, ranked by
signal strength, plus business-impact and qualitative findings), and
`docs/modeling_findings.md` (model comparison, precision/recall@k,
calibration limitations, and the net verdict on what ships) for the full
write-up. README's own findings/decisions summary will be filled in as the
final report pass, to avoid duplicating a moving target.

## What's next

- Write the final ~1-page report/README section synthesizing
  `docs/eda_findings.md` + `docs/modeling_findings.md`
- Address the calibration drift noted in `docs/modeling_findings.md`
  (train/test base-rate shift) via periodic retraining or recalibration,
  if this were to move beyond a one-off analysis
- Stricter point-in-time cutoff based on outcome-*resolution* time
  (`review_answer_timestamp` / `order_delivered_customer_date`) rather than
  purchase time -- documented as a known simplification in
  `src/features/build_features.py`
