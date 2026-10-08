import uuid

from langsmith import Client

from bot import answer

ls = Client()

print("Acme support bot. Ask a question (blank line or 'quit' to exit).")
while True:
    question = input("\nYou: ").strip()
    if question.lower() in {"", "quit", "exit"}:
        break

    run_id = uuid.uuid4()
    result = answer(question, langsmith_extra={"run_id": run_id})
    print(f"\nBot [{result['category']}]: {result['answer']}")

    fb = input("Helpful? (y/n, enter to skip): ").strip().lower()
    if fb in {"y", "n"}:
        # Attaches a thumbs up/down score to the exact trace in LangSmith
        ls.create_feedback(run_id, key="user_score", score=1 if fb == "y" else 0)

ls.flush()
