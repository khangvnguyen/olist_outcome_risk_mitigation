# Problem Framing (v2 — target decision confirmed post-EDA)

This is a living document. Section 1's target decision is now confirmed
(see below). Sections 2-4 remain hypotheses to be finalized as feature
engineering and modeling proceed.

## 1. What is a "bad outcome"?

**DECISION (post-EDA):** `bad_review` (review_score <= 2) is the primary target.
EDA (see `output/eda/summary.md`) showed `is_late` and `is_canceled_or_unavailable`
are only distantly related to `bad_review` -- most bad reviews (66%) occur on
orders that arrived on time, and canceled orders are a small (1.2%), distinct
segment. Blending them into one composite label would obscure more than it
reveals. `is_late` and `is_canceled_or_unavailable` remain useful as
**diagnostic sub-analyses** (why orders go bad) and as the basis for a
point-in-time seller-history feature (seller's historical on-time rate), but
are not part of the target itself.

**Restated problem (post-EDA):** using information available at/near order
time (order attributes, and seller history computed with a point-in-time
cutoff so no future information leaks in), predict whether a given order
will receive a low review score (1-2 stars), so Ops can intervene during
the fulfillment window rather than finding out after the fact. Note this is
closer to "predict at time of dispatch" than "predict at instant of
checkout" in practical terms, since some of the most useful features
(seller history) only stabilize with order volume -- worth stating plainly
rather than overclaiming immediacy.

Candidate operationalisations considered:

| Candidate | Source | Pros | Cons |
|---|---|---|---|
| `review_score <= 2` | `olist_order_reviews_dataset` | Direct proxy for dissatisfaction; business already collects it | Only ~58% of orders have a text comment (score itself has 0% missing though); subjective, confounded by things outside seller/product control (e.g. customer mood) |
| Late delivery: `order_delivered_customer_date > order_estimated_delivery_date` | `olist_orders_dataset` | Objective, operationally actionable (Ops can act on ETA slippage directly) | Doesn't capture product-quality complaints; ~3% missing delivered dates (likely lost/cancelled orders) |
| Order not delivered: `order_status in {canceled, unavailable}` | `olist_orders_dataset` | Objective, severe outcome | Small class (~1.2% combined); may be driven by stock/logistics issues unrelated to "quality" |
| Composite: low review OR late OR canceled | combination | Broadest definition of "bad experience" | Blends causally distinct problems into one target — risks an uninterpretable model |

*(This table reflects the pre-EDA candidate list. Decision confirmed above:
`review_score <= 2` is primary; lateness/cancellation are diagnostic, not
target components. Resolved by the actual EDA numbers, not by the a priori
guess this table represents.)*

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
- `review_score`, `review_comment_*` themselves, and `review_answer_timestamp`
- Final `order_status` if it's later than "processing" states

This boundary needs to be enforced mechanically in the feature pipeline
(explicit allowlist), not just as a design note, since it's easy to
accidentally leak via joins.

**Resolved (post-EDA):** seller-level variance is real but moderate, not
extreme — the worst decile of sellers (by order volume, n>=10) accounts for
~2.5x their proportional share of bad reviews (11.4% of bad reviews from
4.6% of order volume), and there's ~no correlation between seller order
volume and bad-review rate (r=0.006), so "seller experience" isn't a usable
shortcut proxy. This is meaningful but not dominant signal -- worth
including as a feature (with the point-in-time cutoff described above), but
not sufficient on its own; category and price/freight carry comparable
signal (see `output/eda/summary.md` sections 4-6 for the exact,
volume-weighted-std comparison across groupings). The point-in-time cutoff
is worth the implementation complexity given this.

## 3. What "useful" means here

**Decision (post-EDA):** build an order-level classifier (predicting
`bad_review`) using pre-outcome order attributes plus point-in-time seller
history, evaluated primarily on ranking quality (can Ops trust a risk score
to prioritize a limited pool of interventions?) rather than raw accuracy,
given the ~15% positive rate. A ranked seller risk table is a secondary,
cheap-to-produce deliverable alongside it, since seller signal is real but
not concentrated enough to replace a multivariate model on its own (see
section 2). "Useful" = a risk score Ops can act on for a subset of orders
during the fulfillment window, not a guarantee of individual-order
certainty -- precision/recall tradeoffs and calibration will be reported
honestly rather than oversold.

## 4. Known data-quality caveats to carry into EDA

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