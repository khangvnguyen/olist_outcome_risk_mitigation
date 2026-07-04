# Olist Outcome Risk — Predicting Poor Customer Experiences

> Status: complete end-to-end -- data acquisition, EDA, feature engineering,
> modeling. Full depth per stage lives in `docs/`.

## Setup

1. Get the data, either:
   - **Automatic:** create a Kaggle API token (kaggle.com/settings → "Create
     New Token") and place it at `~/.kaggle/kaggle.json`, or
   - **Manual:** download https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
     and unzip the 9 CSVs into `data/raw/`
2. Run `docker compose up --build`

Results land in `output/` (gitignored -- regenerated on each run, not committed).

## Repo structure
```
src/
  data/download.py       # data acquisition (Kaggle API, falls back to manual)
  data/loader.py         # shared raw-loading/joining logic
  eda/run_eda.py         # EDA -> output/eda/
  features/build_features.py  # leakage-safe feature table -> output/features/
  models/train_model.py  # HGB risk model + seller risk table -> output/model/
  pipeline.py            # single entrypoint, orchestrates all stages
docs/
  problem_framing.md     # target definition + feature-leakage design doc
  eda_findings.md        # decision-ready EDA summary feeding the feature plan
  modeling_findings.md   # decision-ready modeling summary
```

## Problem framing

The ask was open-ended: "predict which orders/products/sellers are at risk
of a bad customer experience." I operationalised "bad outcome" as
`bad_review` (`review_score <= 2`, ~14.7% of reviewed orders), deliberately
**not** blending in lateness or cancellation -- EDA showed those are only
distantly related to review score (most bad reviews happen on on-time
orders), so a composite target would obscure more than it reveals. The
resulting task: an order-level risk score, using only information available
at/near order time (no post-outcome leakage), that Ops can use to
**prioritize a limited pool of interventions** during the fulfillment
window -- evaluated on ranking quality, not raw accuracy, since that's what
"useful to a triage team" means at a ~15% positive rate. Full reasoning:
`docs/problem_framing.md`.

## Key findings

- **Signal ranking:** product history > seller history > customer state >
  category > seller state > distance (`docs/eda_findings.md`) -- *who*
  sold/made the product matters far more than distance or seller state.
- **66% of bad reviews happen on orders that arrive on time** -- lateness
  ~6x's the per-order risk (~54% vs. ~9%) but is only a partial lever;
  review text points to product-mismatch/defect complaints as the other
  major driver, which no structured column captures.
- **The model works, modestly:** HistGradientBoostingClassifier reaches
  PR-AUC 0.197 vs. a 0.109 no-model baseline, catching **33% of bad reviews
  (48% of their order value) by flagging the riskiest 20% of orders** --
  beating a "rank by seller history" heuristic only moderately, since that
  heuristic alone already captures most of the signal
  (`docs/modeling_findings.md`).
- **Calibration drifts over time** (train positive rate 15.6% vs. test
  10.9%) -- a real base-rate shift, not a bug. Use the score to *rank*
  orders, not as a literal probability, without periodic retraining.
- **Churn cost is inconclusive with this data** -- Olist's baseline
  repeat-purchase rate (~4%) is too low for a before/after comparison to
  detect an effect. Stated as a limitation, not glossed over.

## Design decisions

- **Leakage discipline enforced mechanically:** an explicit feature
  allowlist asserted in code; seller/product/category "history" features
  are point-in-time (cumulative-prior-to-this-order), not full-dataset
  averages.
- **Product history shrunk toward category rate** (`k=20`) since 95% of
  products have fewer than 10 prior orders -- raw per-product rates would
  mostly be noise.
- **Time-based train/test split**, not random -- honest evaluation for a
  model whose seller/product features are themselves time-dependent.
- **HistGradientBoostingClassifier over XGBoost/LightGBM/logistic
  regression:** native NaN/categorical handling, no new dependency, and it
  beat logistic regression on every axis -- LR's `class_weight="balanced"`
  improved ranking but wrecked calibration, a real tradeoff worth noting.
- **A cheap secondary deliverable** (`output/model/seller_risk_table.csv`):
  a ranked seller risk table needing no model, since seller signal alone
  is real, if not sufficient.

## What's next (with more time or data)

- Retrain/recalibrate periodically to correct the temporal base-rate drift.
- Cut off seller/product history on outcome-*resolution* time (review/
  delivery date) instead of purchase time -- a stated simplification
  (`src/features/build_features.py`).
- Revisit the product-history shrinkage constant -- its contribution was
  smaller than EDA's univariate ranking suggested.
- A causal design (matched cohorts or an experiment) to actually measure
  churn cost, which this one-shot dataset can't support.
- Item-level, multi-seller attribution instead of the "primary
  (highest-price) item" approximation used throughout (~2% of orders).
