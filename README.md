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

Results land in `output/` (gitignored -- regenerated on each run, not committed,
except `output/nlp/` -- see below).

## Repo structure
```
src/
  data/download.py       # data acquisition (Kaggle API, falls back to manual)
  data/loader.py         # shared raw-loading/joining logic
  eda/run_eda.py         # EDA -> output/eda/
  features/build_features.py  # leakage-safe feature table -> output/features/
  models/train_model.py  # HGB risk model + seller risk table -> output/model/
  nlp/categorize_reviews.py  # LLM categorization of negative reviews -> output/nlp/
  pipeline.py            # single entrypoint, orchestrates all stages
docs/
  problem_framing.md     # target definition + feature-leakage design doc
  eda_findings.md        # decision-ready EDA summary feeding the feature plan
  modeling_findings.md   # decision-ready modeling summary
```

### Negative review categorization (`src/nlp/categorize_reviews.py`)

A separate, manually-run stage: it sends the ~1-2 star review comments through
OPENROUTER (`OPENROUTER_API_KEY` required) in chunks, sorting each into one of 8
complaint categories (late delivery, wrong item, damaged product, etc.).
It's **not** wired into `src/pipeline.py` / `docker compose up`, since it's
the only stage with a paid external API dependency:
```
export OPENROUTER_API_KEY=...
python -m src.nlp.categorize_reviews
```
Its two output files (`output/nlp/review_categories.csv`, `output/nlp/summary.md`)
are committed to the repo (a deliberate `.gitignore` exception) so the
results are reviewable without needing your own API key.

## Problem framing

The ask was open-ended: "predict which orders/products/sellers are at risk
of a bad customer experience." I operationalised "bad outcome" as
`bad_review` (`review_score <= 2`, ~14.7% of reviewed orders), deliberately
**not** blending in lateness or cancellation -- EDA showed those are only
distantly related to review score (most bad reviews happen on on-time
orders), so a composite target would obscure more than it reveals.
The resulting task: build an order-level risk score, 
using only information available at/near order time, that Ops can use to prioritize a limited pool
of interventions during the fulfillment window. I report ROC-AUC/PR-AUC as
general ranking diagnostics, but the business-facing evaluation is
precision/recall@k: if Ops can only act on the riskiest 1%, 5%, 10%, or 20%
of orders, how many bad outcomes and how much bad-review order value do we
catch? Full reasoning: `docs/problem_framing.md`.

## Key findings

- **Signal ranking:** product history > seller history > customer state >
  category > seller state > distance (`docs/eda_findings.md`) -- *who*
  sold/made the product matters far more than distance or seller state.
- **66% of bad reviews happen on orders that arrive on time** -- lateness
  ~6x's the per-order risk (~54% vs. ~9%) but is only a partial lever. LLM
  categorization of the complaint text (`src/nlp/categorize_reviews.py`)
  shows those on-time bad reviews are mostly **incomplete orders (25.5%)
  and wrong/divergent items (19.3%)**, not damage or "poor quality" as a
  first word-frequency pass suggested -- a fulfillment-accuracy problem, no
  structured column captures it, and it varies by product category (e.g.
  wrong-item complaints concentrate in home appliances/telephony, damage in
  furniture/audio) (`docs/eda_findings.md`).
- **The model works, modestly:** a HistGradientBoostingClassifier reaches
  PR-AUC 0.197 vs. a 0.109 no-model baseline. More importantly for Ops,
  at the top-20% risk threshold it achieves ~18% precision and ~33% recall,
  capturing **48% of bad-review order value**. It beats a "just rank by
  seller history" heuristic, but only by a moderate margin.
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
  regression/random forest:** native NaN/categorical handling, no new
  dependency, and it beat logistic regression on every axis -- LR's
  `class_weight="balanced"` improved ranking but wrecked calibration, a
  real tradeoff worth noting. Also tried `RandomForestClassifier` as a
  cross-check (same features/split, pre-committed swap criteria) -- it
  didn't beat HGB on PR-AUC or recall@20%, so HGB stays shipped
  (`docs/modeling_findings.md`).
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
- Fold complaint categorization into the regular pipeline and extend it to
  the 25% of bad reviews with no comment text, then use category *mix*
  (not just rate) for ops ticket routing and as a seller/product signal
  that distinguishes fulfillment-accuracy or fraud problems from logistics
  ones -- not usable as a model feature (post-outcome text), but actionable
  on its own (`docs/eda_findings.md`).
