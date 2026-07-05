# EDA Findings

This note summarizes the EDA results that shaped the feature plan and model design. The full generated report is in `output/eda/summary.md`; this file focuses on the conclusions and why they matter.

## Target

The main target is:

```text
bad_review = review_score <= 2
```

This gives a positive rate of about **14.7%**.

Late delivery and cancellations are important diagnostics, but they are not part of the target. They describe different failure modes, and combining them with low reviews would make the model harder to interpret. The target decision is explained in more detail in `docs/problem_framing.md`.

## Main Drivers of Bad Reviews

I compared group-level bad-review rates using volume-weighted standard deviation. In plain terms: groups with larger values show more separation between good and bad outcomes, while giving more weight to groups with enough order volume to trust.

| Grouping | Volume-weighted std | Read |
|---|---:|---|
| Product | ~9.9 pp | Strongest raw signal, but sparse. Only 42% of order volume has at least 10 prior orders per product, so product history needs smoothing. |
| Seller | ~7.2 pp | Second strongest signal. Seller history is more stable because 94% of order volume has at least 10 prior orders per seller. |
| Customer state | ~3.1 pp | Real signal in large states, for example RJ around 21% vs. SP around 13%. Small states at the extremes are mostly noise. |
| Category | ~2.3 pp | Useful, but not dominant. |
| Seller state | ~1.1 pp | Weak. Customer geography matters more than seller geography. |
| Distance | ~0 | Very weak relationship with bad reviews (`r=0.046`). |

I also checked seller-to-customer route effects. The worst lane found was likely just picking up the customer-state effect again, not a reliable route-specific pattern. I did not carry route-level features forward.

## Lateness

Late delivery is a strong risk signal for a single order:

- about **54%** of late orders receive a bad review
- about **9%** of on-time orders receive a bad review

But late delivery explains only part of the problem. Only about 8% of orders are late, so late orders account for about **34%** of all bad reviews. The other **66%** happen on orders that arrive on time.

This means delivery performance matters, but it is not the whole story. Also, `is_late` itself cannot be a model feature because it is only known after delivery. The usable version is historical seller delivery performance, computed only from orders before the order being scored.

## Business Impact

- **Order value:** bad-review orders represent 14.2% of order count but 16.8% of order value. Bad reviews skew slightly toward higher-value orders.
- **Repurchase / churn:** this dataset is not enough to prove churn impact. Olist's repeat-purchase rate is only around 4%, so a simple before/after comparison has very little power. The observed gap is about R$0.16 per customer, which is not meaningful enough to treat as evidence. A churn estimate would need a longer customer history or a causal design.

## What the Review Text Adds

The 66% of bad reviews that were on-time needed more explanation. A word-frequency pass and sample review pointed toward product mismatch or defects: wrong item, damaged item, item different from the photo, and similar issues.

I then ran a separate LLM categorization step on negative reviews with comment text:

- script: `src/nlp/categorize_reviews.py`
- model: `google/gemini-2.5-flash` through OpenRouter
- coverage: 10,830 of 14,484 bad reviews, or 74.8%
- output: `output/nlp/summary.md`

The labels clarified the story:

- **On-time bad reviews are mostly fulfillment-accuracy problems.** `Incomplete Order / Missing Items` is 25.5% and `Wrong Item Delivered / Product Divergence` is 19.3%. Together they make up about 45% of on-time bad reviews.
- **Poor product quality is not the main on-time issue.** `Damaged or Broken Product` is 12.5% and `Poor Quality` is 11.6%.
- **The structured late flag is useful but imperfect.** Of reviews labeled `Late Delivery or Non-Delivery`, only 70.7% were late by the structured `is_late` field. Some customers complain about delivery even when the estimated-date rule says the order was on time.
- **Complaint mix varies by category.** Damage is more common in bulky or fragile categories such as `office_furniture`, `furniture_living_room`, and `audio`. Wrong-item complaints are higher in categories with many similar variants, such as `home_appliances`, `telephony`, and `small_appliances`.
- **Missing-item complaints skew higher value.** Median order value is R$183 for `Incomplete Order / Missing Items` versus R$96 for `Poor Quality / Below Expectations`. This is weak evidence, but it fits the idea that partial short-ships are more likely in larger or multi-item orders.

Review text is post-outcome, so it is not allowed as a predictive model feature. Still, it is useful for root-cause analysis and future Ops routing.

## Feature Plan

Features carried into modeling:

- **Order details:** category, item price, freight value, payment type, installments, customer state, distance, and estimated delivery window
- **Seller history:** smoothed prior bad-review rate, smoothed prior late-delivery rate, and order count to date
- **Product history:** smoothed prior bad-review rate and order count to date
- **Time:** purchase month or season

Seller and product history use point-in-time cutoffs. For each order, the feature only uses orders that happened earlier. This is important because the model is meant to predict risk before the outcome is known.

Product history also needs shrinkage because most products have very little history. The smoothing formula is:

```text
smoothed_rate = (n * product_rate + k * category_rate) / (n + k)
```

This pulls low-volume products toward their category average. Without this, raw product rates would mostly be noise, and using `product_id` directly would risk memorizing rare products.

## Excluded Features

These are excluded because they leak future information or are too weak:

- actual delivery dates
- `is_late`
- final review score or review text
- final order status beyond early states
- seller-to-customer route effects

## Open Items

- **Multi-seller orders:** the current approach uses a primary-item approximation. This affects only about 2% of orders, but a production system should handle item-level attribution more carefully.
- **Churn impact:** the current dataset cannot support a strong churn estimate. This needs a causal study, matched cohorts, or a longer post-period.
- **Review category routing:** complaint categories could help route tickets to logistics, product QA, fraud, or support. This would require folding the categorization step into the regular pipeline and finding a way to handle bad reviews with no comment text.
