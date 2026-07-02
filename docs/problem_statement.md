### Context

You work for a large online marketplace. Poor customer experiences — late deliveries, mismatched expectations, low-quality products — drive negative reviews, refund requests, and churn. The Head of Marketplace Operations has asked:
"We need to get ahead of bad customer outcomes. Can you help us understand which orders are at risk of resulting in a poor experience before they go wrong — or better yet, which product and seller characteristics predict trouble?"

### Your Task
Assess the feasibility of predicting poor customer outcomes and deliver whatever you think is most useful.
You define:
- The problem framing (prediction? root-cause analysis? segmentation? something else?)
- What constitutes a "bad outcome" and how you operationalise it
- The modelling approach (if you choose to model at all)
- The evaluation criteria
- What "useful" means in this context
It is entirely valid to conclude that certain aspects are not feasible with this data — as long as you explain why and what you'd need.

### What You Must Deliver
A GitHub repository containing:
- All code needed to reproduce your results
- A README.md (max ~1 page) covering:
    - Your problem framing and key findings
    - Design decisions and reasoning
    - How to run and reproduce
    - What you'd do next with more time or data
- Dockerised execution: Your solution must run end-to-end via a single command inside a Docker container, saving outputs to an output/ directory
- No Jupyter notebooks as the main artefact (scripts or CLI app)
Your solution should automatically download the dataset (e.g., via the Kaggle API or a scripted download) or clearly document where to place the files. Either approach is fine — just make sure docker compose up (or equivalent) works after one clearly documented setup step.


### What We're Looking For
- Problem Framing: Did you ask the right question? Is your framing useful to the business, not just technically interesting?
- Judgment & Tradeoffs: What did you choose to focus on and what did you deliberately leave out? Can you explain why?
- Depth of Analysis: Did you find anything genuinely interesting in the data? Did you dig beyond surface-level patterns?
- Code Quality: Clear, readable, well-structured. Simple > clever.
- Reproducibility: Does it actually run? Are instructions clear?
- Honesty: Did you acknowledge limitations, uncertainty, and what you don't know?

We care more about how you think than how many techniques you apply. A thoughtful, focused analysis that acknowledges its limitations will score higher than a kitchen-sink approach that claims false precision.
