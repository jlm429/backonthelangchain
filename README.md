![Python](https://img.shields.io/badge/python-3.10%20to%203.13-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Package Manager: Poetry](https://img.shields.io/badge/package%20manager-Poetry-60A5FA.svg)
![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)
![Lint: Ruff](https://img.shields.io/badge/lint-Ruff-261230.svg)
![Workflow: LangGraph](https://img.shields.io/badge/workflows-LangGraph-black.svg)
![Provider: OpenAI](https://img.shields.io/badge/provider-OpenAI-412991.svg)
![RAG: FAISS](https://img.shields.io/badge/RAG-FAISS-orange.svg)
![Optional: Voyage](https://img.shields.io/badge/optional-Voyage-purple.svg)
![Optional: LangSmith](https://img.shields.io/badge/optional-LangSmith-green.svg)

# backonthelangchain

Building, evaluating, and refining LLM-powered systems while demonstrating modern agent-assisted software engineering practices.

Modern AI applications combine software engineering discipline, agentic workflow patterns, evaluation, and tooling to produce reliable systems. This repository explores both the implementation of AI applications and the engineering workflows used to build them.

## Installation

<details>
<summary><strong>Core Installation</strong></summary>

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

</details>

<details>
<summary><strong>Optional RAG Dependencies</strong></summary>

Install the additional dependencies required for the RAG examples:

```bash
poetry install -E rag
```

Required environment variables:

```bash
OPENAI_API_KEY=your_api_key
VOYAGE_API_KEY=your_voyage_api_key
```

</details>

## Agent-Assisted Development

<details>
<summary><strong>Developing with Coding Agents</strong></summary>

This repository is designed for development with modern coding agents, including Codex, Claude Code, Gemini CLI, and future agent-based tools.

Before making changes, review:

- `AGENTS.md` — repository conventions and engineering workflow
- `skills/backonthelangchain/SKILL.md` — project-specific guidance
- `CHANGELOG.md` — user-facing changes

Recommended workflow:

```text
Create feature branch
        │
        ▼
Read AGENTS.md and repository Skill
        │
        ▼
Implement changes
        │
        ▼
Run ./scripts/check.sh
        │
        ▼
Update docs / CHANGELOG (if needed)
        │
        ▼
Commit changes
        │
        ▼
(Optional) Validate with no-mistakes
        │
        ▼
Push branch
        │
        ▼
Open Pull Request
```

</details>

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
       Top 3 FAQ Chunks
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
