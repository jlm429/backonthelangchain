# backonthelangchain

Welcome to **backonthelangchain**, a personal sandbox for exploring advanced LangChain concepts, including:

- **Agentic AI** (multi-step, decision-based reasoning)
- **RAG pipelines** (retrieval-augmented generation)
- **LangGraph** (stateful and structured agent flows)

This repo showcases experimental and practical use cases for building smarter LLM applications using the LangChain framework and related tools.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`github_lookup.ipynb`](github_lookup.ipynb) | Uses Tavily and OpenAI to search for a GitHub user, retrieve their public profile, and summarize it using an LLM + Pydantic output parser. |
| [`react_news_summary_agent.ipynb`](react_news_summary_agent.ipynb) | A ReAct-based LangChain agent that fetches recent news on a topic, summarizes each article, and estimates reading level and time using custom tools. |
---

## 🛠️ Requirements

- Python 3.10+
- LangChain
- OpenAI API key
- Tavily API key
- Jupyter

