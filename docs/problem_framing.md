# Problem Framing (Draft v2 — after initial EDA)

This is still a living document, but the first EDA pass in
`output/eda/summary.md` now gives enough evidence to tighten the framing before
modeling.

## 1. What is a "bad outcome"?

Candidate operationalisations, from the tables available:

| Candidate | Source | Pros | Cons |
|---|---|---|---|
| `review_score <= 2` | `olist_order_reviews_dataset` | Direct proxy for dissatisfaction; business already collects it; review score exists for 99.2% of orders | Subjective, confounded by things outside seller/product control (e.g. customer mood); text comments are much sparser than scores |
| Late delivery: `order_delivered_customer_date > order_estimated_delivery_date` | `olist_orders_dataset` | Objective, operationally actionable (Ops can act on ETA slippage directly) | Doesn't capture product-quality complaints; ~3% missing delivered dates (likely lost/cancelled orders) |
| Order not delivered: `order_status in {canceled, unavailable}` | `olist_orders_dataset` | Objective, severe outcome | Small class (~1.2% combined); may be driven by stock/logistics issues unrelated to "quality" |
| Composite: low review OR late OR canceled | combination | Broadest definition of "bad experience" | Blends causally distinct problems into one target — risks an uninterpretable model |

**Decision after EDA:** use `review_score <= 2` as the primary target. It has a
14.68% positive rate among reviewed orders, which is large enough for a
classifier or risk-ranking model to be feasible.

Late delivery and canceled/unavailable status are not model inputs for a
pre-outcome model. They remain diagnostic outcomes:
- late delivered orders have a much higher bad-review rate (53.99% vs. 9.23%)
- but on-time delivered orders still contribute about two thirds of bad reviews
  among orders with both review and lateness known
- canceled/unavailable orders are rare (1.24%) but severe, with average review
  score 1.67 where reviewed

This means the story is not "predict lateness and we are done." Logistics is a
major risk amplifier, while product/category, seller quality, and expectation
effects still matter for the majority of bad reviews.

## 2. Feature eligibility: pre-outcome vs. post-outcome

Because the business ask is explicitly "at risk of resulting in a poor
experience **before** it goes wrong," any feature only known after the
order is placed and progressing must be excluded from a predictive model, or
we're not predicting anything — we're describing the past.

**Eligible ("known at/near order time"):**
- Product: category, price, weight/dimensions, photo count, description length
- Seller: historical review score / late-delivery rate *as of before this order*, tenure, state, catalog size
- Order: price, freight value, payment type/installments, customer state, estimated delivery window (`order_estimated_delivery_date - order_purchase_timestamp`), distance proxy (customer zip vs seller zip via geolocation)
- Time: purchase day-of-week/month/season

**Ineligible (leakage — only known after the fact):**
- `order_delivered_carrier_date`, `order_delivered_customer_date` (actual, not estimated)
- `is_late`, `days_late`, and actual `delivery_days`
- `review_score`, `review_comment_*` themselves, and `review_answer_timestamp`
- Final `order_status` if it's later than "processing" states

This boundary needs to be enforced mechanically in the feature pipeline
(explicit allowlist), not just as a design note, since it's easy to
accidentally leak via joins.

**Post-EDA seller feature decision:** seller history is worth trying, but only
with a point-in-time cutoff. In single-seller reviewed orders, 619 sellers have
at least 30 orders, and seller bad-review rates vary widely. The top 5% of
sellers by bad-review count account for 56.7% of bad reviews but also 51.4% of
orders, so seller concentration is not a standalone explanation. Seller identity
should be represented as historical risk features rather than used as a naive
post-hoc league table.

## 3. What "useful" means here

Useful means ranking orders by risk early enough for Marketplace Operations to
act, while also explaining the main operational levers:
- order-level bad-review risk score using only features known at or near order
  time
- interpretation by feature groups: seller history, category/product, price and
  freight, customer/seller geography, and payment structure
- separate diagnostic readout for late and canceled/unavailable orders, because
  these explain important failure modes but are not valid model inputs

The next deliverable should therefore be a simple, honest classifier or risk
ranker plus a short segmentation readout. A pure seller ranking would be too
narrow, because most bad reviews are not explained by late delivery alone and
seller bad-review concentration is partly just seller order volume.

## 4. Known data-quality caveats to carry forward

- `olist_geolocation_dataset` has ~26% duplicate rows and multiple lat/lng
  per zip prefix — needs de-duplication/aggregation (e.g. centroid per
  zip prefix) before using it for distance features.
- `product_category_name` has ~1.85% missing, and category needs joining
  through `product_category_name_translation` to get English labels.
- Order items table is one-row-per-item, not one-row-per-order — needs
  explicit aggregation (sum price/freight, count sellers, count distinct
  products) before joining to order-level target.
- `payment_installments` has a min of 0, which is unexpected (should be
  >=1) — worth checking as a data quality flag.
