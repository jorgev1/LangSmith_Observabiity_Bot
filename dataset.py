from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

DATASET_NAME = "support-bot-eval"

# (question, expected category, expected FAQ doc id or None if unanswerable)
CASES = [
    ("Can I get my money back if I bought an annual plan last week?", "billing", "refunds"),
    ("Where do I find past invoices?", "billing", "invoices"),
    ("If I upgrade mid-month will I be charged the full price?", "billing", "plan-change"),
    ("My card got declined, what happens now?", "billing", "failed-payments"),
    ("I forgot my password and the reset link isn't working", "account", "reset-password"),
    ("I lost my phone with my authenticator app and my backup codes", "account", "two-factor"),
    ("How many people can I add to the Team plan?", "account", "invite-teammates"),
    ("I want to close my account permanently", "account", "delete-account"),
    ("I keep getting 429 errors from your API", "technical", "rate-limits"),
    ("How often should I rotate API keys?", "technical", "api-keys"),
    ("Is the service down right now?", "technical", "status"),
    ("Can I download everything in my workspace as JSON?", "technical", "data-export"),
    ("Do you offer a 99.9% uptime guarantee on the Team plan?", "other", "sla"),
    ("Which browsers do you support?", "technical", "browsers"),
    # Unanswerable: the bot should say it doesn't know
    ("Do you integrate with Salesforce?", "other", None),
    ("What's your phone number for support?", "other", None),
    ("Can I pay with cryptocurrency?", "billing", None),
    ("Do you have an Android app?", "other", None),
]


def main():
    client = Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset '{DATASET_NAME}' already exists, nothing to do.")
        return
    ds = client.create_dataset(DATASET_NAME, description="Support bot Q&A with expected category and source doc")
    client.create_examples(
        dataset_id=ds.id,
        inputs=[{"question": q} for q, _, _ in CASES],
        outputs=[
            {"category": cat, "doc_id": doc, "answerable": doc is not None}
            for _, cat, doc in CASES
        ],
    )
    print(f"Created '{DATASET_NAME}' with {len(CASES)} examples.")


if __name__ == "__main__":
    main()
