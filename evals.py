from dotenv import load_dotenv

load_dotenv()

import anthropic
from langsmith import evaluate

from bot import ANTHROPIC_KWARGS, FAST_MODEL, PROMPT_VERSION, answer, is_refusal, text_of
from dataset import DATASET_NAME

judge = anthropic.Anthropic(**ANTHROPIC_KWARGS)  # unwrapped so judge calls don't clutter the experiment traces


def target(inputs: dict) -> dict:
    return answer(inputs["question"])


def category_match(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "category_match", "score": int(outputs["category"] == reference_outputs["category"])}


def retrieval_hit(outputs: dict, reference_outputs: dict) -> dict:
    if not reference_outputs["answerable"]:
        return {"key": "retrieval_hit", "score": 1, "comment": "n/a (unanswerable)"}
    return {"key": "retrieval_hit", "score": int(reference_outputs["doc_id"] in outputs["doc_ids"])}


def refusal_correct(outputs: dict, reference_outputs: dict) -> dict:
    # Answerable questions should be answered; unanswerable ones should be refused.
    refused = is_refusal(outputs["answer"])
    return {"key": "refusal_correct", "score": int(refused == (not reference_outputs["answerable"]))}


def groundedness(outputs: dict) -> dict:
    # LLM-as-judge: is every claim in the answer supported by the retrieved context?
    if is_refusal(outputs["answer"]):
        return {"key": "groundedness", "score": 1, "comment": "refusal"}
    context = "\n".join(outputs["context"]) or "(none)"
    resp = judge.messages.create(
        model=FAST_MODEL,
        max_tokens=5,
        messages=[
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context}\n\nANSWER:\n{outputs['answer']}\n\n"
                    "Is every factual claim in the ANSWER supported by the CONTEXT? "
                    "Reply with only YES or NO."
                ),
            }
        ],
    )
    return {"key": "groundedness", "score": int(text_of(resp).upper().startswith("YES"))}


if __name__ == "__main__":
    evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[category_match, retrieval_hit, refusal_correct, groundedness],
        experiment_prefix=f"support-bot-{PROMPT_VERSION}",
        metadata={"prompt_version": PROMPT_VERSION},
        max_concurrency=4,
    )
