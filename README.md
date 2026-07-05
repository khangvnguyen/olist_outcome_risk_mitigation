# Olist Outcome Risk

This project asks a practical Ops question: **which orders are most likely to end in a poor customer experience, while there is still time to act?**

I define a poor outcome as a 1-2 star review (`bad_review`). Late delivery and cancellations matter, but I keep them out of the target because they describe different problems. Most bad reviews in this dataset are not simply late orders.

## Key Findings

- About **14.7%** of reviewed orders receive a 1-2 star review.
- **Late orders are risky**, with about 54% bad reviews versus about 9% for on-time orders. But late delivery explains only part of the problem: **66% of bad reviews happen on orders that arrive on time**.
- Product and seller history are the strongest signals in EDA. Customer state and category help; seller state and distance add much less.
- Negative review text suggests many on-time bad reviews are about **missing items** or **wrong products**, not just poor product quality.
- The final model is useful as a **ranking tool**, not a perfect predictor. On the held-out time split, the HistGradientBoosting model gets PR-AUC **0.197** versus **0.109** for the no-model baseline. If Ops reviews the riskiest 20% of orders, it catches about **33%** of bad reviews and **48%** of bad-review order value.
- Raw probabilities should not be treated as exact. The bad-review rate drops over time in the data, so the model needs retraining or recalibration before production use.

More detail is in:

- `docs/problem_framing.md`
- `docs/eda_findings.md`
- `docs/modeling_findings.md`

## Design Choices

- **Target:** 1-2 star review. It is the clearest direct signal of customer dissatisfaction.
- **Prediction timing:** use only data available at or near order time. Actual delivery dates, final review text, and review scores are excluded from model features.
- **Evaluation:** focus on precision/recall at the top risk bands, because Ops can only act on a limited number of orders.
- **Split:** use a time-based train/test split instead of a random split, since this is closer to real deployment.
- **Features:** include order details, category, price/freight, customer state, estimated delivery window, and point-in-time seller/product history.
- **Model:** ship `HistGradientBoostingClassifier`. It performs best overall while keeping dependencies simple. I also include a seller risk table because it is easy to explain and useful even without the model.

## How to Run

First, make the Kaggle data available in one of two ways:

1. Automatic: create a Kaggle API token and place it at `~/.kaggle/kaggle.json`. My file is as below: {"username":"khangng*****","key":<REDACTED>}
2. Manual: download the Olist dataset from Kaggle and unzip the 9 CSV files into `data/raw/`.

Then run:

```bash
docker compose up --build
```

The pipeline writes regenerated outputs to `output/`, including EDA tables, the feature table, model metrics, and the seller risk table.

The review-text categorization step is separate because it uses a paid external API. Its outputs are already committed under `output/nlp/` so the analysis can be reviewed without an API key.

## What I Would Do Next

- Retrain or recalibrate the model on a rolling schedule to handle the time drift.
- Test different product-history smoothing, since raw product signal is strong but sparse.
- Use review categories for Ops routing, for example separating late delivery from wrong-item or missing-item complaints.
- Improve attribution for the small share of multi-seller orders.
- Run a causal or experimental study to measure churn impact. This dataset alone is not enough to prove it.
