# Modeling Findings

This note summarizes the modeling results and the final model choice. The full generated metrics are in `output/model/metrics.md`.

All numbers below are from the held-out, time-based test split:

- test set: 19,735 orders
- test bad-review rate: 10.90%
- train bad-review rate: 15.62%

That train/test rate gap matters. It means the model should be treated as a ranking tool unless it is retrained or recalibrated.

## Model Comparison

| Model | ROC-AUC | PR-AUC | Brier score |
|---|---:|---:|---:|
| constant baseline | 0.500 | 0.109 | 0.099 |
| seller heuristic | 0.592 | 0.156 | 0.097 |
| logistic regression | 0.608 | 0.172 | 0.217 |
| hist gradient boosting | 0.620 | 0.197 | 0.095 |
| random forest | 0.613 | 0.189 | 0.096 |

The main question was whether a full model adds enough value over a simple seller-history ranking.

The answer is **yes, but not by a huge margin**. HistGradientBoosting improves PR-AUC from 0.156 to 0.197. That is a real gain, but it also shows that the seller-only table is useful on its own.

## Business View: Precision and Recall at Top Risk Bands

Ops will not act on every order. The useful question is: if the team reviews the highest-risk 1%, 5%, 10%, or 20% of orders, how many bad outcomes do we catch?

| Top risk band | Model | Precision | Recall | Bad-review order value captured |
|---|---|---:|---:|---:|
| 1% | seller heuristic | 0.305 | 0.028 | R$11.7k (2.8%) |
| 1% | logistic regression | 0.365 | 0.034 | **R$38.7k (9.4%)** |
| 1% | hist gradient boosting | **0.518** | **0.047** | R$19.8k (4.8%) |
| 5% | seller heuristic | 0.219 | 0.100 | R$46.3k (11.2%) |
| 5% | logistic regression | **0.274** | **0.126** | **R$89.6k (21.7%)** |
| 5% | hist gradient boosting | 0.264 | 0.121 | R$81.0k (19.6%) |
| 10% | seller heuristic | 0.176 | 0.161 | R$87.8k (21.2%) |
| 10% | logistic regression | **0.217** | **0.199** | **R$136.2k (32.9%)** |
| 10% | hist gradient boosting | 0.210 | 0.193 | R$124.9k (30.2%) |
| 20% | seller heuristic | 0.151 | 0.277 | R$139.9k (33.8%) |
| 20% | logistic regression | 0.169 | 0.310 | R$190.7k (46.1%) |
| 20% | hist gradient boosting | **0.182** | **0.333** | **R$199.6k (48.2%)** |

The model choice depends a little on capacity:

- At the tightest 1% cutoff, HistGradientBoosting has the best precision: 51.8%.
- At 5% and 10%, logistic regression is slightly better on precision, recall, and value captured.
- At 20%, HistGradientBoosting is best on precision, recall, and bad-review value captured.

For this project I ship HistGradientBoosting because it is the best all-around choice. It has the strongest PR-AUC, the best Brier score among the learned models, and the best 20% operating point. Still, if Ops could only review 1-5% of orders, I would re-check the model choice against the exact intervention goal.

## Why Not Logistic Regression?

Logistic regression ranks orders reasonably well, but its probabilities are poorly calibrated.

Its Brier score is **0.217**, worse than the constant baseline at **0.099**. The reason is `class_weight="balanced"`: it helps ranking on an imbalanced target, but it pushes predicted probabilities away from the true bad-review rate.

That tradeoff matters here. I want the shipped model to rank well without producing probability scores that are obviously misleading. Logistic regression stays useful as a comparison point, but it is not the final model.

## Random Forest Check

I also tested `RandomForestClassifier` as a second non-linear baseline.

The pre-set rule was: replace HistGradientBoosting only if random forest improved PR-AUC by at least 0.015, kept recall@20%, and kept Brier score at or below 0.105.

It did not meet that bar:

- PR-AUC: 0.189 vs. 0.197 for HistGradientBoosting
- recall@20%: 0.314 vs. 0.333
- Brier score: 0.096 vs. 0.095

Random forest is close, but it does not change the final choice.

## What Drives the Model?

Permutation importance on the test set points to the following order:

| Feature | Importance |
|---|---:|
| `seller_bad_review_rate_smoothed` | 0.044 |
| `total_freight_value` | 0.020 |
| `primary_category` | 0.006 |
| `total_item_price` | 0.004 |
| `estimated_delivery_days` | 0.004 |
| `product_bad_review_rate_smoothed` | 0.002 |
| `seller_late_rate_smoothed` | 0.002 |

Everything else is near noise level, including `customer_state` and `customer_seller_distance_km`.

The biggest shift from EDA is product history. Product-level bad-review rate looked very strong on its own, but it contributes much less after smoothing and after category is already in the model.

The likely reason is coverage. Most products have very few prior orders, so the smoothed product feature gets pulled toward the category average. This is safer than overfitting rare products, but it also reduces the product feature's independent signal.

Practical read: **seller history, category, freight, and price carry most of the model.** Product-level tracking may still be useful, but it needs better treatment before it earns much complexity.

## Calibration

The model overstates risk across the score range. For example, the top decile predicts about 30.8% bad reviews but realizes about 21.0%.

This is not just a modeling bug. The train period has a 15.6% bad-review rate, while the test period has 10.9%. The base rate changed over time.

For deployment, this means:

- use the score to rank orders
- avoid treating the score as an exact probability
- retrain or recalibrate on a rolling schedule before production use

## Seller Risk Table

The pipeline also outputs `output/model/seller_risk_table.csv`.

This table is not point-in-time and is not a replacement for the order-level model, but it is useful for Ops. It ranks sellers by smoothed bad-review rate and surfaces the same high-risk sellers seen in EDA, including sellers such as `4342d4b2...` and `b1b39487...`.

The seller table is cheap, easy to explain, and actionable even if the classifier is not deployed yet.

## Final Recommendation

- Ship the HistGradientBoosting score as an **order ranking tool**.
- Do not present its raw score as a calibrated probability.
- Include the seller risk table as a simple secondary deliverable.
- Revisit calibration, product-history smoothing, and exact model choice once the real Ops capacity and intervention cost are known.
