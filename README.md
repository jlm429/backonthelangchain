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
| [`ica_rag_pipeline.ipynb`](ica_rag_pipeline.ipynb) | A LangChain + Pinecone pipeline that ingests ICA content from PDFs and YouTube, then answers user questions using retrieval-augmented generation. |
| [`langgraph_reflection_agent.ipynb`](langgraph_reflection_agent.ipynb) | An iterative tweet revision loop using LangGraph. Simulates alternating tweet generation and critique using ChatOpenAI and a branching message graph. |
| [`reflexion_agent.ipynb`](reflexion_agent.ipynb) | Implements a Reflexion-style research agent using LangGraph and Tavily. The agent answers a question, reflects on its response, and revises using real-time search. |
| [`agentic_rag_flow.ipynb`](agentic_rag_flow.ipynb) | A dynamic RAG pipeline using LangGraph with document grading, hallucination detection, and adaptive routing. Inspired by Self-RAG (Asai et al., 2023) and Adaptive-RAG (Jeong et al., 2024). |

---

## 🛠️ Requirements

- Python 3.10+
- LangChain
- OpenAI API key
- Tavily API key
- Jupyter

