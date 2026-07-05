"""
LLM-based categorization of negative review comments (see src/data/loader.py
for the order-level table this reads from).

Run:
    export OPENROUTER_API_KEY=...   # https://aistudio.google.com/apikey
    python -m src.nlp.categorize_reviews

Outputs (under output/nlp/):
    review_categories.csv          order_id, review_score, review_comment_message,
                                    category_id, category_name -- one row per
                                    negative review with comment text
    summary.md                     category distribution + example comments
    review_categories_partial.jsonl  resumability checkpoint (gitignored, not
                                      a deliverable -- safe to delete to force
                                      a full re-run)

Scope: only reviews with review_score <= 2 and non-empty
review_comment_message -- matches the ad hoc extraction previously done in
Untitled.ipynb / negative_reviews.txt (that dump discarded order_id, so its
results couldn't be joined back to the order table; this does keep it).
Each review gets exactly ONE category (single label), chosen by the model
from the 8 categories below.

Standalone stage, not part of src/pipeline.py: unlike every other stage,
this one makes paid external API calls and needs a credential
(OPENROUTER_API_KEY) that isn't available in CI/docker compose, so it's run
manually, once, and its two output files above ARE committed to the repo
(see the .gitignore exception) so results are reviewable without an API key.
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

from src.data.loader import RAW_DIR, load_raw, build_order_level_table

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "nlp"

MODEL_NAME = "google/gemini-2.5-flash"
BATCH_SIZE = 40                # reviews per API call
SLEEP_BETWEEN_CALLS_SEC = 4.0  # stay under OpenRouter rate limits
RETRY_BACKOFF_SEC = 15.0

UNCATEGORIZED_ID = 0  # sentinel: batch/parse failure, not a real category

CATEGORIES = [
    {
        "id": 1,
        "name": "Late Delivery or Non-Delivery",
        "name_pt": "Atraso ou Não Entrega",
        "description": "The customer did not receive the product within the promised "
                        "timeframe, the tracking is frozen, or the item never arrived at all.",
        "keywords": ["não recebi", "atraso", "demorou", "correios", "não chegou"],
    },
    {
        "id": 2,
        "name": "Incomplete Order / Missing Items",
        "name_pt": "Pedido Incompleto / Itens Faltando",
        "description": "The order arrived, but it was missing units (e.g. ordered 2, "
                        "received 1), specific accessories, or parts needed for assembly/functioning.",
        "keywords": ["faltando", "só veio uma unidade", "incompleto", "veio pela metade"],
    },
    {
        "id": 3,
        "name": "Wrong Item Delivered / Product Divergence",
        "name_pt": "Produto Incorreto / Divergente",
        "description": "The customer received a completely different item, the wrong "
                        "color, the wrong model, the wrong clothing size, or an incompatible "
                        "voltage (e.g. 220V instead of 110V).",
        "keywords": ["produto errado", "diferente", "cor errada", "voltagem", "trocado"],
    },
    {
        "id": 4,
        "name": "Damaged or Broken Product",
        "name_pt": "Produto Danificado ou Quebrado",
        "description": "The item arrived physically damaged, broken, scratched, cracked, "
                        "torn, or stopped working on the very first day due to poor transit "
                        "handling or physical defects.",
        "keywords": ["quebrado", "riscado", "com defeito", "arranhado", "rachando"],
    },
    {
        "id": 5,
        "name": "Counterfeit or Misleading Advertising",
        "name_pt": "Produto Falsificado / Propaganda Enganosa",
        "description": "The ad claimed the item was 'original', but the user received a "
                        "pirate copy/replica. Also includes cases where features shown in "
                        "the photos/description are completely absent (e.g. non-functional buttons).",
        "keywords": ["falsificado", "réplica", "pirata", "não é original", "propaganda enganosa"],
    },
    {
        "id": 6,
        "name": "Poor Quality / Below Expectations",
        "name_pt": "Baixa Qualidade / Produto Inferior",
        "description": "The product arrived intact and matches the description, but the "
                        "materials are cheap, flimsy, poorly finished, or don't live up to "
                        "the buyer's quality expectations.",
        "keywords": ["péssima qualidade", "material ruim", "fraco", "transparente", "mal acabado"],
    },
    {
        "id": 7,
        "name": "Customer Support & Refund Issues",
        "name_pt": "Problemas de Atendimento e Reembolso",
        "description": "Difficulty contacting the store/marketplace, ignored emails, "
                        "trouble activating a warranty/return, or delays getting a credit "
                        "card chargeback after cancellation.",
        "keywords": ["não respondem", "tentar contato", "estorno", "cancelamento", "SAC", "justiça"],
    },
    {
        "id": 8,
        "name": "General Complaints Without Insights",
        "name_pt": "Reclamações Gerais sem Contexto",
        "description": "Very short, low-value phrases where the user expresses anger or "
                        "dissatisfaction but doesn't explicitly state what went wrong.",
        "keywords": ["ruim", "péssimo", "odiei", "não recomendo"],
    },
]

VALID_CATEGORY_IDS = {c["id"] for c in CATEGORIES}
CATEGORY_NAME_BY_ID = {c["id"]: c["name"] for c in CATEGORIES}
CATEGORY_NAME_BY_ID[UNCATEGORIZED_ID] = "Uncategorized (parse error)"


class ReviewCategoryResult(BaseModel):
    order_id: str
    category_id: int


def _category_prompt_block() -> str:
    lines = []
    for cat in CATEGORIES:
        lines.append(
            f"{cat['id']}. {cat['name']} ({cat['name_pt']})\n"
            f"   Description: {cat['description']}\n"
            f"   Keywords: {', '.join(cat['keywords'])}"
        )
    return "\n".join(lines)


SYSTEM_INSTRUCTION = (
    "You are categorizing Brazilian e-commerce (Olist) customer reviews written in "
    "Portuguese. Every review below is from an order with a low star rating (1-2 "
    "stars). Assign EXACTLY ONE category id (1-8) to each review, choosing the single "
    "best-fit category even if a review could arguably touch more than one. If a "
    "review is a short, vague complaint with no specific reason given, use category 8.\n\n"
    "Categories:\n" + _category_prompt_block()
)


def _build_prompt(batch: list) -> str:
    reviews_block = "\n".join(
        json.dumps({"order_id": row["order_id"], "text": row["text"]}, ensure_ascii=False)
        for row in batch
    )
    return (
        SYSTEM_INSTRUCTION
        + "\n\n"
        + """

    For EACH of the following reviews, return one object with that same
    order_id and the chosen category_id.

    Return ONLY valid JSON.
    Your entire response must be a single JSON object.
    Do not output any text before or after the JSON.

    The JSON must follow exactly this schema:

    Format:

    {
    "results": [
        {
        "order_id": "...",
        "category_id": 1
        }
    ]
    }

    Do not include explanations.
    Do not use markdown.
    Do not wrap in ```.

    Return exactly one object for every review.

    """
        + reviews_block
    )

def _call_openrouter(client, batch):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": _build_prompt(batch),
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    # Expecting:
    # {
    #   "results": [
    #      {"order_id":"...", "category_id":1},
    #      ...
    #   ]
    # }

    return [
        ReviewCategoryResult(**item)
        for item in data["results"]
    ]


def categorize_batch(client, batch):
    """Categorize one batch, retrying once on API/parse failure. Reviews
    missing from the response, or tagged with a category id outside 1-8, are
    marked with the UNCATEGORIZED_ID sentinel rather than dropped -- every
    order_id in `batch` always gets a result row."""
    for attempt in (1, 2):
        try:
            parsed = _call_openrouter(client, batch)
            by_id = {r.order_id: r.category_id for r in parsed}
            out = []
            n_bad = 0
            for row in batch:
                cat_id = by_id.get(row["order_id"])
                if cat_id not in VALID_CATEGORY_IDS:
                    n_bad += 1
                    cat_id = UNCATEGORIZED_ID
                out.append({"order_id": row["order_id"], "category_id": cat_id})
            if n_bad:
                print(f"[nlp] WARNING: {n_bad}/{len(batch)} reviews in batch missing/invalid "
                      f"category_id, marked uncategorized.")
            return out
        except Exception as exc:
            print(f"[nlp] WARNING: batch call failed (attempt {attempt}/2): {exc}")
            if attempt == 1:
                time.sleep(RETRY_BACKOFF_SEC)

    print(f"[nlp] WARNING: batch of {len(batch)} reviews failed twice, marking all uncategorized.")
    return [{"order_id": row["order_id"], "category_id": UNCATEGORIZED_ID} for row in batch]


def write_summary(df: pd.DataFrame, path: Path):
    lines = [
        "# Negative Review Categorization Summary\n",
        "Auto-generated by `src/nlp/categorize_reviews.py`. Do not hand-edit.\n",
        f"Model: `{MODEL_NAME}`. Scope: reviews with `review_score <= 2` and non-empty "
        "`review_comment_message` from the order-level table (`src/data/loader.py`). "
        "Single label per review (the model picks the single best-fit category).\n",
        f"Total categorized: {len(df):,}\n",
        "## Category distribution\n",
        "| Category | Count | % |",
        "|---|---|---|",
    ]
    counts = df["category_name"].value_counts()
    for name, count in counts.items():
        lines.append(f"| {name} | {count:,} | {100 * count / len(df):.1f}% |")

    lines.append("\n## Example comments per category\n")
    for cat_id in sorted(df["category_id"].dropna().unique()):
        name = CATEGORY_NAME_BY_ID.get(cat_id, "Uncategorized (parse error)")
        examples = df.loc[df["category_id"] == cat_id, "review_comment_message"].head(3).tolist()
        lines.append(f"### {name}\n")
        for ex in examples:
            lines.append(f"- \"{ex}\"")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[nlp] ERROR: OPENROUTER_API_KEY environment variable is not set. "
              "Get a key at https://openrouter.ai/keys and export it "
              "(e.g. `export OPENROUTER_API_KEY=...`).")
        return 1

    print("[nlp] Loading raw tables and building order-level table...")
    raw = load_raw(RAW_DIR)
    df = build_order_level_table(raw)

    negative = df[
        (df["review_score"] <= 2)
        & df["review_comment_message"].notna()
        & (df["review_comment_message"].str.strip() != "")
    ][["order_id", "review_score", "review_comment_message"]].reset_index(drop=True)
    print(f"[nlp] {len(negative):,} negative reviews (review_score <= 2) with comment text.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = OUTPUT_DIR / "review_categories_partial.jsonl"

    processed = {}
    if partial_path.exists():
        with open(partial_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    processed[rec["order_id"]] = rec["category_id"]
        print(f"[nlp] Resuming: {len(processed):,} reviews already checkpointed in {partial_path.name}.")

    remaining = negative[~negative["order_id"].isin(processed.keys())]
    print(f"[nlp] {len(remaining):,} reviews remaining to categorize.")

    if len(remaining) > 0:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        n_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
        with open(partial_path, "a", encoding="utf-8") as f:
            for i in range(n_batches):
                chunk = remaining.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                batch = [
                    {"order_id": row.order_id, "text": row.review_comment_message}
                    for row in chunk.itertuples()
                ]
                results = categorize_batch(client, batch)
                for r in results:
                    f.write(json.dumps(r) + "\n")
                    processed[r["order_id"]] = r["category_id"]
                f.flush()
                print(f"[nlp] Batch {i + 1}/{n_batches} done ({len(processed):,}/{len(negative):,} total).")
                if i < n_batches - 1:
                    time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    print("[nlp] Assembling final output...")
    negative["category_id"] = negative["order_id"].map(processed)
    negative["category_name"] = negative["category_id"].map(CATEGORY_NAME_BY_ID)

    out_path = OUTPUT_DIR / "review_categories.csv"
    negative.to_csv(out_path, index=False)
    write_summary(negative, OUTPUT_DIR / "summary.md")

    print(f"[nlp] Done. See {out_path} and {OUTPUT_DIR / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
