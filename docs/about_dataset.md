## About Dataset

### Brazilian E-Commerce Public Dataset by Olist

Welcome! This is a Brazilian ecommerce public dataset of orders made at Olist Store. The dataset has information of **100k orders from 2016 to 2018** made at multiple marketplaces in Brazil. Its features allows viewing an order from multiple dimensions: from order status, price, payment and freight performance to customer location, product attributes and finally reviews written by customers. We also released a geolocation dataset that relates Brazilian zip codes to lat/lng coordinates.

This is real commercial data, it has been anonymised, and references to the companies and partners in the review text have been replaced with the names of **Game of Thrones** great houses.

### Join it With the Marketing Funnel by Olist

We have also released a Marketing Funnel Dataset. You may join both datasets and see an order from Marketing perspective now!

Instructions on joining are available on this Kernel.

---

## Context

This dataset was generously provided by Olist, the largest department store in Brazilian marketplaces. Olist connects small businesses from all over Brazil to channels without hassle and with a single contract. Those merchants are able to sell their products through the Olist Store and ship them directly to the customers using Olist logistics partners. See more on our website: [www.olist.com](https://www.olist.com)

After a customer purchases the product from Olist Store a seller gets notified to fulfill that order. Once the customer receives the product, or the estimated delivery date is due, the customer gets a satisfaction survey by email where he can give a note for the purchase experience and write down some comments.

> ### ⚠️ Attention
> 
> 
> * An order might have multiple items.
> * Each item might be fulfilled by a distinct seller.
> * All text identifying stores and partners where replaced by the names of Game of Thrones great houses.
> 
> 

---

## Data Schema

The data is divided in multiple datasets for better understanding and organization. Please refer to the following data schema when working with it:

```text
                  +-------------------------------+                 +------------------------+
                  |  olist_order_payments_dataset |                 | olist_products_dataset |
                  +-------------------------------+                 +------------------------+
                                  ▲                                              ▲
                                  │ order_id                                     │ product_id
                                  ▼                                              ▼
+----------------------------+  order_id  +----------------------+  order_id  +--------------------------+  seller_id  +-----------------------+
| olist_order_reviews_datst  | ◄─────────►| olist_orders_dataset |◄──────────►| olist_order_items_datst  |◄───────────►| olist_sellers_dataset |
+----------------------------+            +----------------------+            +--------------------------+             +-----------------------+
                                  ▲                                                                                    ▲
                                  │ customer_id                                                                        │ zip_code_prefix
                                  ▼                                                                                    ▼
                                +------------------------------+            zip_code_prefix             +---------------------------+
                                | olist_order_customer_dataset |◄──────────────────────────────────────►| olist_geolocation_dataset |
                                +------------------------------+                                        +---------------------------+
```

### Classified Dataset

We had previously released a classified dataset, but we removed it at Version 6. We intend to release it again as a new dataset with a new data schema. While we don't finish it, you may use the classified dataset available at the Version 5 or previous.

---

## Inspiration

Here are some inspiration for possible outcomes from this dataset.

* **NLP:** This dataset offers a supreme environment to parse out the reviews text through its multiple dimensions.
* **Clustering:** Some customers didn't write a review. But why are they happy or mad?
* **Sales Prediction:** With purchase date information you'll be able to predict future sales.
* **Delivery Performance:** You will also be able to work through delivery performance and find ways to optimize delivery times.
* **Product Quality:** Enjoy yourself discovering the products categories that are more prone to customer insatisfaction.
* **Feature Engineering:** Create features from this rich dataset or attach some external public information to it.

---

## Acknowledgements

Thanks to Olist for releasing this dataset.

Here is a structured markdown description of `image_ab9d23.png` designed specifically to help an LLM contextualize the data relationships (such as multiple sellers, freight, and listings) within the Olist ecosystem.

You can drop this section directly into your Markdown file under a new heading:

---

## Example of a Product Listing

### Product Metadata

* **Product Title:** Smartphone Motorola Moto G6 Play Dual Chip Android Oreo - 8.0 Tela 5.7" Octa-Core 1.4 GHz 32GB 4G Câmera 13MP - Índigo
* **Product ID / Code (Cód.):** 133453169
* **Review Rating:** 4 out of 5 stars (based on 215 reviews)

### Marketplace Dynamics (Multi-Seller Buy Box)

The interface showcases the marketplace concept where multiple distinct merchants sell the exact same item profile:

* **Selected Offer (Olist Buy Box):**
* **Seller:** Sold and delivered by **olist** (`olist_sellers_dataset`).
* **Price:** R$ 1.299,00 (with an option for installment plans up to 10x of R$ 129,90 without interest).
* **Logistics / Freight:** Freight cost is R$ 26,04 with a delivery estimate of 7 to 10 business days (`olist_orders_dataset` / freight performance).
* **Stock Alert:** "Corra! Temos apenas 5 no estoque" (Hurry! We only have 5 left in stock).


* **Alternative Seller Offers:**
* **Seller 2:** Price R$ 1.069,90 | Freight: R$ 38,32 | Delivery: 7 to 10 business days.
* **Seller 3:** Price R$ 975,00 | Freight: R$ 22,94 | Delivery: 5 to 6 business days.
* **Other Options:** Indicates additional merchant options starting from R$ 959,00.


### Additional Features Shown

* **Cross-Selling / Bundling:** An add-on checkbox offers an "ANKER SoundCore Bluetooth 12W Speaker" for an additional + R$ 429,99.
* **Omnichannel / Pickup:** A banner reading *"pegue na loja hoje!"* highlights a local pickup option (tying back to geolocation data).