# EDA Findings Summary (post-EDA, pre-modeling)

Consolidated from `output/eda/summary.md` (auto-generated, full detail) plus
the interpretation/back-and-forth in `docs/problem_framing.md`. This doc is
the decision-ready reference for the feature engineering and modeling
stages — it states conclusions and why, not raw numbers (see the auto-report
for those).

## Target (confirmed)

`bad_review` = review_score <= 2. ~14.7% positive rate. `is_late` and
`is_canceled_or_unavailable` are diagnostic, not part of the target (see
`problem_framing.md` section 1 for why).

## What actually predicts bad_review — ranked by signal strength

Measured consistently via volume-weighted std of bad_review_rate across
groups (comparable across groupings, see `output/eda/summary.md` sections
4/5/9):

| Grouping | Volume-weighted std | Note |
|---|---|---|
| Product | ~9.9 pp | Strongest signal found, but only 42% of order volume has enough history (>=10 orders) per product to trust it -- needs shrinkage, see below |
| Seller | ~7.2 pp | Second strongest; 94% of volume has enough history (>=10 orders); no seller-experience effect (volume vs. rate correlation ~0) |
| Customer state | ~3.1 pp | Real signal on large-n states (RJ ~21% vs. SP ~13%, both n>10k -- credible). Small-n states (RR, AP) at the "worst/best" extremes are noise, not signal -- don't lead with those. |
| Category | ~2.3 pp | Real but modest |
| Seller state | ~1.1 pp | Weak. Customer-side geography matters more than seller-side. |
| Distance (km) | ~0 (r=0.046 with bad_review) | Not worth much modeling complexity |

**Seller-state -> customer-state "lane" effects:** the worst lane found
(PR->CE) is likely just re-detecting the customer-state effect (CE is a
mid-table state) rather than a genuine route-specific effect. Not pursued
further -- customer state alone is the more defensible feature.

## Lateness (diagnostic, not a feature)

Late delivery is a strong per-order risk multiplier (~54% bad_review rate
when late vs. ~9% on-time) but explains only ~34% of total bad reviews,
since only ~8% of orders are late. **The majority of bad reviews (66%)
happen on orders that arrived on time** -- delivery performance is a real
but partial lever. `is_late` itself is leakage (only known post-delivery)
and cannot be a model input; the actionable version is a seller's
**historical** on-time rate (point-in-time cutoff), used as a proxy.

## Business impact

- **Revenue at risk:** bad-review orders account for a slightly
  *higher* share of order value (16.8%) than their share of order count
  (14.2%) -- bad reviews skew mildly toward higher-value orders. Small but
  consistent with the price/freight finding in section 6.
- **Repurchase/churn:** inconclusive by design, not by finding. Olist's
  baseline repeat-purchase rate is very low (~4%) across the board, which
  leaves little statistical room for a first-order-experience effect to
  show up in a simple before/after comparison (observed gap ~R$0.16/customer,
  not distinguishable from noise). **Honest conclusion: this dataset/method
  cannot demonstrate a churn cost of bad reviews** -- would need either a
  matched/causal design or a longer post-period than this snapshot allows.
  Not pursued further; stated as a limitation, not glossed over.

## Qualitative signal (bad reviews on on-time orders)

Word-frequency and sample review of the 66% "unexplained by lateness"
segment (`output/eda/tables/bad_ontime_review_word_frequency.csv` and
`..._sample.csv`) point toward product-mismatch/defect complaints (wrong
item, damaged, different from photo) rather than logistics. This is
consistent with -- and helps explain *why* -- product-level signal (above)
is the strongest grouping found. Not usable as a model feature (text is
post-outcome), but supports prioritizing product-level features and
explains the ceiling on how much delivery-focused fixes alone can help.

## Feature plan for modeling (carried into src/features/)

**Include, with point-in-time cutoffs (only using history prior to the
order being scored) for seller and product features specifically:**
- Order-level: category, item price, freight value, payment type,
  installments, customer state, distance (low expected value, cheap to
  include)
- Seller history: smoothed historical bad-review rate, smoothed historical
  late-delivery rate, order volume-to-date
- Product history: smoothed historical bad-review rate (shrunk toward
  category rate by sample size -- see reasoning below), order volume-to-date
- Time: purchase month/season

**Product-level feature requires shrinkage, not raw identity:**
`smoothed_rate = (n*product_rate + k*category_rate) / (n+k)` for some
smoothing constant k (e.g. 10-20 orders). Reasoning: 95% of the product
catalogue has <10 prior orders, so raw per-product rates are mostly noise,
and using raw `product_id` as a categorical directly (32,951 levels) risks
memorization and breaks entirely on unseen products at inference. Same
shrinkage logic optionally applies to seller history, though seller
coverage is high enough (94%) that it matters less there.

**Explicitly excluded (leakage):** `is_late`, actual delivery dates,
`order_status` beyond early states, review score/text, seller-state
lane effects (too weak / redundant with customer state alone).

## Open items not pursued (explicitly out of scope for now)

- Multi-seller order attribution (currently approximated by assigning the
  whole order's outcome to every seller/product involved; ~2% of orders)
- True causal estimate of churn cost (would need a different study design)
- Deeper NLP on review text beyond word frequency (out of scope given text
  can't be a model feature anyway; useful for root-cause narrative only)