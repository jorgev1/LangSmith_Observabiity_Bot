from dotenv import load_dotenv

load_dotenv()

import json
import os
from pathlib import Path

import anthropic
from langsmith import traceable
from langsmith.wrappers import wrap_anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Only needed if your API key isn't scoped to a workspace (set ANTHROPIC_WORKSPACE_ID in .env)
_workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")
ANTHROPIC_KWARGS = (
    {"default_headers": {"anthropic-workspace-id": _workspace_id}} if _workspace_id else {}
)

# wrap_anthropic auto-traces every messages.create call as an LLM span
client = wrap_anthropic(anthropic.Anthropic(**ANTHROPIC_KWARGS))

FAST_MODEL = "claude-haiku-4-5-20251001"  # classification + judging
MAIN_MODEL = "claude-sonnet-5-5"  # answer generation
CATEGORIES = ["billing", "technical", "account", "other"]
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")

SYSTEM_PROMPTS = {
    "v1": (
        "You are Acme Cloud's support assistant. Answer ONLY using the CONTEXT provided. "
        "If the context does not contain the answer, reply exactly: "
        "\"I don't have that information. Please contact support@acme.example.\" "
        "Keep answers under 4 sentences and cite source ids in square brackets, e.g. [refunds]."
    ),
    # Deliberately weak: no grounding rule. Used to demo a regression caught by evals.
    "v2_loose": (
        "You are a friendly, helpful support assistant for Acme Cloud. "
        "Answer the customer's question as helpfully as you can."
    ),
}

FAQ = json.loads(Path(__file__).with_name("faq.json").read_text())
_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
_matrix = _vectorizer.fit_transform([f"{d['title']} {d['text']}" for d in FAQ])


def is_refusal(text: str) -> bool:
    return "don't have that information" in text.replace("\u2019", "'").lower()


def text_of(resp) -> str:
    """Join the text blocks of a response, skipping thinking and other block types."""
    return "".join(b.text for b in resp.content if b.type == "text").strip()


@traceable(run_type="chain", name="classify")
def classify(question: str) -> str:
    resp = client.messages.create(
        model=FAST_MODEL,
        max_tokens=10,
        system=(
            "Classify the customer question into exactly one of: "
            + ", ".join(CATEGORIES)
            + ". Reply with the single lowercase label only."
        ),
        messages=[{"role": "user", "content": question}],
    )
    label = text_of(resp).lower()
    return label if label in CATEGORIES else "other"


@traceable(run_type="retriever", name="retrieve")
def retrieve(question: str, k: int = 3) -> list[dict]:
    scores = cosine_similarity(_vectorizer.transform([question]), _matrix)[0]
    top = scores.argsort()[::-1][:k]
    # page_content + metadata is the shape LangSmith renders as retrieved documents
    return [
        {
            "page_content": f"{FAQ[i]['title']}: {FAQ[i]['text']}",
            "metadata": {"id": FAQ[i]["id"], "score": float(scores[i])},
        }
        for i in top
        if scores[i] > 0
    ]


@traceable(run_type="chain", name="generate")
def generate(question: str, docs: list[dict]) -> str:
    context = (
        "\n".join(f"[{d['metadata']['id']}] {d['page_content']}" for d in docs)
        or "(no relevant documents found)"
    )
    resp = client.messages.create(
        model=MAIN_MODEL,
        max_tokens=2000,  # headroom in case the model emits a thinking block first
        system=SYSTEM_PROMPTS[PROMPT_VERSION],
        messages=[{"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}],
    )
    return text_of(resp)


@traceable(
    name="support_bot",
    tags=["support-bot"],
    metadata={"prompt_version": PROMPT_VERSION},
)
def answer(question: str) -> dict:
    category = classify(question)
    docs = retrieve(question)
    reply = generate(question, docs)
    return {
        "answer": reply,
        "category": category,
        "doc_ids": [d["metadata"]["id"] for d in docs],
        "context": [d["page_content"] for d in docs],
    }
