from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from typing import List, TypedDict, Dict, Any, Literal
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnableSequence
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph

load_dotenv()

# Load and split documents
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Optional vectorstore ingestion
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=OpenAIEmbeddings(),
#     persist_directory="./.chroma",
# )

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()

class GraphState(TypedDict, total=False):
    question: str
    generation: str
    web_search: bool
    documents: List[Document]

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

# Grader for document relevance
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing relevance of a retrieved document to a user question."),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])
retrieval_grader = grade_prompt | structured_llm_grader

def grade_documents(state: GraphState) -> Dict[str, Any]:
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def truncate_documents(documents, max_chars=12000):
    total = 0
    result = []
    for d in documents:
        if total + len(d.page_content) > max_chars:
            break
        result.append(d)
        total += len(d.page_content)
    return result

prompt = hub.pull("rlm/rag-prompt")
generation_chain = prompt | llm | StrOutputParser()

def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state.get("documents", [])
    trimmed_docs = truncate_documents(documents)
    generation = generation_chain.invoke({"context": trimmed_docs, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    tavily_results = web_search_tool.invoke({"query": question})
    if isinstance(tavily_results, list) and isinstance(tavily_results[0], dict):
        joined = "\n".join([r.get("content", "") for r in tavily_results])
    elif isinstance(tavily_results, list):
        joined = "\n".join(tavily_results)
    else:
        joined = str(tavily_results)
    return {"documents": [Document(page_content=joined)], "question": question}

class GradeAnswer(BaseModel):
    binary_score: bool = Field(description="Answer addresses the question, 'yes' or 'no'")

answer_grader = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing whether an answer addresses / resolves a question"),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
]) | llm.with_structured_output(GradeAnswer)

class GradeHallucinations(BaseModel):
    binary_score: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

hallucination_grader = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts."),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
]) | llm.with_structured_output(GradeHallucinations)

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "websearch"] = Field(...)

question_router = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at routing a user question to a vectorstore or web search."),
    ("human", "{question}"),
]) | llm.with_structured_output(RouteQuery)

def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    print("---ROUTE QUESTION TO RAG---")
    return "retrieve"

def decide_to_generate(state: GraphState) -> str:
    print("---ASSESS GRADED DOCUMENTS---")
    return "websearch" if state["web_search"] else "generate"

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    if hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]}).binary_score:
        print("---GENERATION IS GROUNDED---")
        if answer_grader.invoke({"question": state["question"], "generation": state["generation"]}).binary_score:
            print("---GENERATION IS USEFUL---")
            return "useful"
        print("---GENERATION NOT USEFUL---")
        return "not useful"
    print("---GENERATION NOT GROUNDED---")
    return "not supported"

RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)
workflow.set_conditional_entry_point(route_question, {
    WEBSEARCH: WEBSEARCH,
    RETRIEVE: RETRIEVE,
})
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, {
    WEBSEARCH: WEBSEARCH,
    GENERATE: GENERATE,
})
workflow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question, {
    "not supported": GENERATE,
    "useful": END,
    "not useful": WEBSEARCH,
})
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)
app = workflow.compile()

print("Hello Advanced RAG")
print(app.invoke({
    "question": "what is gluten-free pizza?",
    "documents": [],
    "generation": "",
    "web_search": False
}))
