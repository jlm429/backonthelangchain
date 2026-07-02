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

...
</details>

<details>
<summary><strong>Safety-Gated Support Router</strong></summary>

...
</details>

<details>
<summary><strong>Safety-Gated Support Router with RAG</strong></summary>

...
</details>
