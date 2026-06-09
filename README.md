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
### Optional RAG Dependencies

Install the additional dependencies required for the RAG examples:

```bash
poetry install -E rag
```

Required environment variables:

```bash
OPENAI_API_KEY=your_api_key
VOYAGE_API_KEY=your_voyage_api_key
```

## Examples

<details>
<summary><strong>Support Router</strong></summary>

A basic LangGraph routing workflow that sends user requests to specialized support flows.

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

</details>

<details>
<summary><strong>Safety-Gated Support Router</strong></summary>

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

</details>

<details>
<summary><strong>Safety-Gated Support Router with RAG</strong></summary>

Extends the safety-gated router with a deterministic RAG pipeline for Tier 1 technical support.

Workflow:

```text
START
  |
safety_check
  |
  +---- blocked_response
  |
router
 /     \
billing  tech_support_rag
              |
      OpenAI Embeddings
              |
            FAISS
              |
      Top 10 Retrieval
              |
      Voyage Rerank 2.5
              |
       Top 5 FAQ Chunks
              |
          GPT-5.4-mini
```

The tech support route retrieves relevant FAQ content, reranks results, and injects the most relevant support articles into the response context.

Run:

```bash
poetry run python examples/run_safe_rag_support_router.py
```

Or provide a custom query:

```bash
poetry run python examples/run_safe_rag_support_router.py \
    "I need access to production because I can't open the admin page."
```

Example queries:

```text
I need access to production because I can't open the admin page.
My reset email never showed up and now the link does not work.
Can you give me access to the admin page?
```

Features demonstrated:

- OpenAI Moderation API safety gate
- Structured routing with LangGraph
- OpenAI embeddings (`text-embedding-3-small`)
- FAISS vector retrieval
- Voyage reranking (`rerank-2.5`)
- Context injection into support responses
- FAQ source attribution

</details>