# LangSmith_Observabiity_Bot
This guide walks participants through building and instrumenting a small customer-support assistant for a fictional product, Acme Cloud. The assistant uses Claude models, and participants then use LangSmith to trace every step, score the results with evaluations, and collect user feedback.

# Observable Support Bot (Claude + LangSmith)

A small customer-support assistant for a fictional product ("Acme Cloud"), built on Claude and instrumented with [LangSmith](https://smith.langchain.com) for tracing, evaluation, and user feedback. It is designed as a hands-on demo and workshop project.

```
question -> classify (Haiku) -> retrieve (TF-IDF over faq.json) -> generate (Sonnet) -> cited answer
```

Each question becomes one parent trace (`support_bot`) with three child spans: `classify`, `retrieve` (shown as retrieved documents), and `generate`. An LLM-as-judge step runs only during evaluations.

## Why different models per step

- **Classify** is a simple labeling task that runs on every question, so it uses the small, fast model (Claude Haiku 4.5).
- **Retrieve** uses plain TF-IDF search. No LLM is needed.
- **Generate** is the step the customer reads, so it uses the stronger model (Claude Sonnet 5.5).
- **Judge** (evals only) is a yes/no groundedness check, so it uses Haiku again.

Traces show latency, tokens, and cost per span, and experiments show whether quality held when you change a model or prompt.

## Prerequisites

- Python **3.10 or newer** (3.12 recommended)
- An [Anthropic API key](https://console.anthropic.com) created **inside a workspace**, with API credits
- A [LangSmith](https://smith.langchain.com) account and API key (the free Developer plan is enough)

## Quick start

### Windows (PowerShell)

```powershell
git clone [GitHub repository URL]
cd [repository folder]

winget install --id=astral-sh.uv -e      # one-time; reopen PowerShell afterward
uv python install 3.12
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
Copy-Item .env.example .env
```

If activation fails with "running scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### macOS / Linux

```bash
git clone [GitHub repository URL]
cd [repository folder]

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Configure

Edit `.env` and fill in your keys:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Your workspace-scoped Anthropic API key |
| `ANTHROPIC_WORKSPACE_ID` | Optional. Only needed if your key is not scoped to a workspace (`wrkspc_...`) |
| `LANGSMITH_TRACING` | `true` to send traces |
| `LANGSMITH_API_KEY` | Your LangSmith API key |
| `LANGSMITH_PROJECT` | Project name for traces, e.g. `support-bot` |
| `PROMPT_VERSION` | `v1` (grounded, default) or `v2_loose` (deliberately weak, for the regression demo) |

### Smoke test

```bash
python -c "from bot import answer; print(answer('How do I get a refund?'))"
```

Then open LangSmith, go to Tracing, open your project, and confirm a `support_bot` trace appears with three child spans.

## Usage

**Browser chat** (with thumbs up/down feedback logged to the trace):

```bash
streamlit run app.py
```

Opens at http://localhost:8501. Restart it after changing `PROMPT_VERSION`.

**Terminal chat:**

```bash
python main.py
```

**Create the evaluation dataset** (18 examples, 4 of which are unanswerable; run once):

```bash
python dataset.py
```

**Run an experiment:**

```bash
python evals.py
```

### Evaluators

| Evaluator | Type | Checks |
|---|---|---|
| `category_match` | Code | Classifier label equals the expected category |
| `retrieval_hit` | Code | Expected FAQ entry was retrieved (unanswerable questions pass automatically) |
| `refusal_correct` | Code | Answerable questions are answered; unanswerable ones are refused |
| `groundedness` | LLM judge (Haiku) | Every claim is supported by the retrieved context |

### Regression demo

Compare the grounded prompt against the deliberately weak one:

```powershell
# Windows PowerShell
$env:PROMPT_VERSION = "v2_loose"; python evals.py
$env:PROMPT_VERSION = "v1"
```

```bash
# macOS / Linux
PROMPT_VERSION=v2_loose python evals.py
```

In LangSmith, open Datasets & Experiments, select both experiments, and use the compare view. Results vary between runs, so check what you actually observe.

## Project structure

```
.
├── bot.py             # classify, retrieve, generate, and the traced answer()
├── app.py             # Streamlit browser chat with feedback buttons
├── main.py            # terminal chat with feedback
├── dataset.py         # creates the LangSmith evaluation dataset
├── evals.py           # runs an experiment with four evaluators
├── faq.json           # 14-entry knowledge base
├── requirements.txt
├── .env.example
└── .gitignore
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `'Anthropic' object has no attribute 'completions'` | Newer Anthropic SDK is incompatible with the LangSmith wrapper | `pip install "anthropic<1"` (already pinned in `requirements.txt`) |
| 400 error: API key is not scoped to a workspace | Key created without a workspace | Create a key inside a workspace, or set `ANTHROPIC_WORKSPACE_ID` in `.env` |
| `'ThinkingBlock' object has no attribute 'text'` | A response starts with a thinking block | Already handled by `text_of()` in `bot.py`; make sure you have the latest code |
| No traces in LangSmith | Missing env vars or wrong workspace selected | Check `LANGSMITH_TRACING=true`, the API key, and the project name |
| Env var syntax fails on Windows | `VAR=value cmd` is bash syntax | Use `$env:VAR = "value"; cmd` in PowerShell |

## Costs

You pay for Anthropic API usage (pay-as-you-go). A full 18-example experiment is typically a few cents, and the free LangSmith Developer plan includes a monthly trace allowance with a 14-day retention for base traces. Prices and limits change, so check the current [Anthropic](https://www.anthropic.com/pricing) and [LangSmith](https://www.langchain.com/pricing) pricing pages.

## Notes

- The retriever uses TF-IDF to keep the demo free of extra dependencies. An embeddings-based retriever is a natural upgrade.
- The category label is displayed and evaluated but not used for routing.
- A workshop guide (Word) accompanies this project. Add it under `docs/` if you want it in the repository.

## License

[Choose a license and add a LICENSE file before publishing.]
