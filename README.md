# Olist Outcome Risk — Predicting Poor Customer Experiences

> Status: repo skeleton + data acquisition stage only. EDA, modeling, and
> report stages are not yet implemented (tracked in `docs/problem_framing.md`).

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
  data/download.py   # data acquisition (Kaggle API, falls back to manual)
  eda/                # (TODO)
  features/           # (TODO)
  models/             # (TODO)
  pipeline.py          # single entrypoint, orchestrates all stages
docs/
  problem_framing.md  # target definition + feature-leakage design doc
output/                # all results land here
```

## Problem framing, key findings, design decisions

Not yet written — pending EDA. See `docs/problem_framing.md` for the current
working hypotheses on target definition and feature scope.

## What's next

- Run EDA against real data to validate/revise target definition
- Decide: order-level classifier vs. seller/product-level segmentation
- Build feature pipeline with explicit pre-outcome allowlist
- Model + evaluate
- Fill in this README's findings/decisions/next-steps sections properly
