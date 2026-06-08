![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Ruff](https://img.shields.io/badge/lint-ruff-blue.svg)
![LangGraph](https://img.shields.io/badge/built%20with-LangGraph-black.svg)

# backonthelangchain

A repository for building and evaluating LLM-powered applications with LangGraph and related technologies.

Modern AI applications live somewhere between hard-coded workflows and fully autonomous agents. They combine traditional software engineering principles with language models to balance flexibility, control, and reliability. This repository explores that space through implementations, prototypes, and reference architectures that emphasize modularity and maintainability.

## Installation

Clone the repository and install dependencies using Poetry:

```bash
git clone https://github.com/jlm429/backonthelangchain.git
cd backonthelangchain

poetry install
```

Create a local `.env` file:

```bash
OPENAI_API_KEY=your_api_key

# Optional
LANGSMITH_API_KEY=your_langsmith_key
```

## Examples

### Support Router

A basic LangGraph routing workflow that sends user requests to specialized support flows.

Pattern demonstrated:

```text
START
  |
router
 /     \
tech   billing
```

Run:

```bash
poetry run python examples/run_support_router.py
```

Or provide a custom query:

```bash
poetry run python examples/run_support_router.py \
    "I was charged twice this month."
```

Example queries:

```text
I cannot log in after enabling MFA.
I was charged twice this month.
```

---

### Safety-Gated Support Router

Extends the router workflow with a pre-router safety check using OpenAI's moderation API.



```text
START
  |
safety_check
  |
  +---- blocked_response
  |
router
 /     \
tech   billing
```

Run:

```bash
poetry run python examples/run_safe_support_router.py
```

Or provide a custom query:

```bash
poetry run python examples/run_safe_support_router.py \
    "I cannot log in after enabling MFA."
```

Example queries:

```text
I hate your support team. They are worthless idiots.
I was charged twice this month.
```

