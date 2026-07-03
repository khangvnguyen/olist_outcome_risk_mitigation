# Modeling Findings (post-modeling, pre-report)

Consolidated from `output/model/metrics.md` (auto-generated, full detail)
plus the design decisions in `docs/problem_framing.md` section 3. This doc
is the decision-ready reference for the final README/report write-up --
states conclusions and why, not raw numbers (see the auto-report for
those). Numbers below are all on the held-out, time-based test split
(19,735 orders, positive rate 10.90%; train positive rate 15.62% -- see
"Calibration" below for why that gap matters).

## Model comparison (confirmed)

| model | roc_auc | pr_auc | brier_score |
|---|---|---|---|
| constant_baseline | 0.500 | 0.109 | 0.099 |
| seller_heuristic (rank by seller history alone) | 0.592 | 0.156 | 0.097 |
| logistic_regression | 0.608 | 0.172 | **0.217** |
| hist_gradient_boosting | 0.620 | 0.197 | 0.095 |

**Verdict on problem_framing.md's open question ("does a multivariate model
earn its complexity over a seller-only heuristic?"): yes, but modestly, not
decisively.** HGB improves PR-AUC over the seller heuristic by ~0.04 (0.197
vs. 0.156) -- a real, consistent gain, but this is not a case where the
simple heuristic turns out to be worthless. A team with limited engineering
capacity could ship the seller risk table alone (see below) and capture
most of the signal.

## Precision / Recall at top-k% risk (the business-facing number)

| k% | model | precision | recall | bad-review order value captured |
|---|---|---|---|---|
| 5% | seller_heuristic | 0.219 | 0.100 | R$46.3k (11.2%) |
| 5% | logistic_regression | **0.274** | **0.126** | **R$89.6k (21.7%)** |
| 5% | hist_gradient_boosting | 0.264 | 0.121 | R$81.0k (19.6%) |
| 10% | seller_heuristic | 0.176 | 0.161 | R$87.8k (21.2%) |
| 10% | logistic_regression | **0.217** | **0.199** | **R$136.2k (32.9%)** |
| 10% | hist_gradient_boosting | 0.210 | 0.193 | R$124.9k (30.2%) |
| 20% | seller_heuristic | 0.151 | 0.277 | R$139.9k (33.8%) |
| 20% | logistic_regression | 0.169 | 0.310 | R$190.7k (46.1%) |
| 20% | hist_gradient_boosting | **0.182** | **0.333** | **R$199.6k (48.2%)** |

**Honest nuance, not smoothed over:** no single model wins at every
operating point. Logistic regression is narrowly *better* than HGB at the
tightest 5%/10% cutoffs; HGB only pulls ahead once Ops can act on a larger
20% pool. If the real intervention capacity turns out to be very small
(e.g. a 5-person Ops team that can only chase ~5% of orders), this would be
worth re-checking rather than assuming HGB is uniformly best.

## Why logistic regression isn't the one to ship despite the table above

LR's Brier score (0.217) is **worse than the constant baseline** (0.099),
even though its ranking metrics look competitive. Cause: `class_weight="balanced"`
was used to counter the ~15% positive rate, which measurably helps ranking
but pushes `predict_proba` outputs away from the true prevalence --
a textbook ranking-vs-calibration tradeoff. This is exactly why
problem_framing.md commits to reporting calibration as its own axis rather
than inferring it from AUC: a model can look good on one and be unusable on
the other. **HGB is the one to ship** -- it's competitive-to-best on every
axis (ROC-AUC, PR-AUC, Brier, and the 20% cutoff) without needing a
calibration-distorting class weight; LR stays in the pipeline only as an
interpretability cross-check.

## What drives the model (permutation importance, HGB, test set)

Ranked: `seller_bad_review_rate_smoothed` (0.044) >> `total_freight_value`
(0.020) > `primary_category` (0.006) > `total_item_price` (0.004) >
`estimated_delivery_days` (0.004) > `product_bad_review_rate_smoothed`
(0.002) > `seller_late_rate_smoothed` (0.002) > everything else (noise
level, including `customer_state` and `customer_seller_distance_km`).

**Interpretation:** broadly consistent with the EDA driver ranking (seller
signal is strong) but with one notable shift -- **product-level history
contributes far less than EDA's raw univariate ranking suggested**
(product had the largest volume-weighted std in EDA section 4/5/9). Most
likely explanation: with k=20 shrinkage toward category (needed because
95% of products have <10 prior orders), the smoothed product feature mostly
collapses into what `primary_category` already captures, leaving little
independent signal once category is already in the model. Practical
implication: **seller history + category + freight/price are the
load-bearing features**; the extra engineering complexity of product-level
tracking is not earning its keep at current shrinkage/coverage levels. A
next step worth trying (not done here): a smaller shrinkage constant or a
product-embedding approach for the ~5% of products with real history,
rather than shrinking all products uniformly.

## Calibration: a real limitation, not a bug to silently fix

Predicted probabilities run above actual rates across every decile (e.g.
top decile predicts 30.8% but realizes 21.0%). Root cause: the train-period
bad_review rate (15.6%) is meaningfully higher than the test-period rate
(10.9%) -- a genuine temporal decline already visible in EDA section 8's
monthly trend, not a modeling error. **Practical implication: if deployed
as-is, raw HGB probability outputs would systematically overstate risk in
the current period.** The model should be used as a *ranking* device
(which is what's evaluated and recommended -- precision/recall@k), not read
as a literal probability, until it's retrained on more recent data or
wrapped in a periodic recalibration step. Flagged as a next step, not
silently absorbed into "the model works."

## Seller risk table cross-check (secondary deliverable)

`output/model/seller_risk_table.csv` (full-dataset, non-point-in-time,
same k=20 shrinkage) surfaces the same worst offenders as EDA's
`output/eda/tables/worst_20_sellers.csv` (e.g. sellers `4342d4b2...`,
`b1b39487...` appear at the top of both). Consistent signal across two
independently-computed views gives confidence the shrinkage isn't
distorting the ranking -- this table is cheap, requires no model, and is
directly actionable for Ops on its own.

## Net verdict / what ships

- **Primary deliverable:** the HGB risk score, used and evaluated as a
  *ranking* tool (precision/recall@k), not a calibrated probability.
- **Secondary deliverable:** the current seller risk table -- cheap,
  interpretable, and directly actionable independent of the classifier.
- **Explicitly not pursued (stated, not hidden):** recalibration/rolling
  retraining to fix the temporal drift; a stricter outcome-resolution-time
  cutoff (already flagged as a simplification in
  `src/features/build_features.py`); hyperparameter tuning beyond sklearn
  defaults + early stopping; alternative product-level shrinkage schemes.
