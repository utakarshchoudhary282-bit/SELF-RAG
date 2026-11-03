from typing import TypedDict, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from config import get_chroma_collection, get_llm

# =====================================================================
# 1. State Schema
# =====================================================================
class GraphState(TypedDict):
    """
    State tracking data flowing between LangGraph nodes.
    Tracks queries, retrieved & graded documents, retries, and final answer.
    """
    question: str                           # Original user question
    query_type: str                         # Classification: conceptual, how-to, troubleshooting, api_reference
    rewritten_query: str                    # Search query optimized for vector store
    documents: List[Dict[str, Any]]          # Retrieved raw chunks
    relevant_documents: List[Dict[str, Any]] # Chunks filtered as relevant by the grader
    retry_count: int                        # Number of rewrite & re-retrieve retries attempted
    max_retries: int                        # Maximum allowed retries before fallback
    answer: str                             # Generated answer
    sources: List[str]                      # Document sources cited
    trace: List[str]                        # Step-by-step node execution log

# =====================================================================
# 2. Graph Nodes
# =====================================================================

def node_query_analysis(state: GraphState) -> Dict[str, Any]:
    """
    Node 1: Query Analysis
    Rewrites and expands the user's raw question to improve retrieval.
    Also classifies the query type (conceptual, how-to, troubleshooting, api_reference).
    """
    question = state["question"]
    retry_count = state.get("retry_count", 0)
    trace = list(state.get("trace", []))
    llm = get_llm()

    if retry_count == 0:
        # First attempt: rewrite question and classify query type
        prompt = (
            "You are a technical query optimizer. Given a user question about technical documentation, "
            "do two things:\n"
            "1. Classify the query type as one of: [conceptual, how-to, troubleshooting, api_reference].\n"
            "2. Rewrite the query with clear technical keywords and synonyms to improve semantic retrieval.\n\n"
            "Format your output exactly as:\n"
            "TYPE: <type>\n"
            "QUERY: <rewritten_query>\n\n"
            f"User Question: {question}"
        )
    else:
        # Retry attempt: previous retrieval had no relevant chunks, broaden/rephrase
        prompt = (
            "The previous search query did not find relevant documents in the technical documentation. "
            f"Please rewrite and broaden this question with alternative technical synonyms for better vector search: {question}\n\n"
            "Format your output exactly as:\n"
            "TYPE: how-to\n"
            "QUERY: <broader_rewritten_query>"
        )

    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        query_type = "how-to"
        rewritten_query = question

        for line in response.splitlines():
            if line.upper().startswith("TYPE:"):
                query_type = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("QUERY:"):
                rewritten_query = line.split(":", 1)[1].strip()
    except Exception as e:
        query_type = "how-to"
        rewritten_query = question

    log_msg = f"Query Analysis (Attempt {retry_count + 1}): type='{query_type}', search_query='{rewritten_query}'"
    trace.append(log_msg)

    return {
        "query_type": query_type,
        "rewritten_query": rewritten_query,
        "trace": trace
    }

def node_retrieval(state: GraphState) -> Dict[str, Any]:
    """
    Node 2: Retrieval
    Searches ChromaDB vector store for top-k most relevant chunks.
    """
    search_query = state.get("rewritten_query", state["question"])
    trace = list(state.get("trace", []))
    collection = get_chroma_collection()

    # Query top 4 chunks
    results = collection.query(
        query_texts=[search_query],
        n_results=4,
        include=["documents", "metadatas", "distances"]
    )

    documents = []
    if results and results.get("documents") and results["documents"][0]:
        docs_list = results["documents"][0]
        meta_list = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs_list)
        for text, meta in zip(docs_list, meta_list):
            documents.append({
                "content": text,
                "source": meta.get("source", "unknown") if meta else "unknown"
            })

    log_msg = f"Retrieval: fetched {len(documents)} chunks from ChromaDB"
    trace.append(log_msg)

    return {
        "documents": documents,
        "trace": trace
    }

def node_document_grader(state: GraphState) -> Dict[str, Any]:
    """
    Node 3: Document Grading (Self-Correction)
    Evaluates each retrieved chunk using an LLM to check if it is truly relevant.
    Filters out irrelevant chunks.
    """
    question = state["question"]
    documents = state.get("documents", [])
    trace = list(state.get("trace", []))
    llm = get_llm()

    relevant_documents = []

    for idx, doc in enumerate(documents):
        prompt = (
            "You are a document grader assessing relevance of a technical documentation chunk to a user's question.\n"
            f"Question: {question}\n"
            f"Document chunk:\n{doc['content']}\n\n"
            "Does this chunk contain information relevant to answering the question? "
            "Respond ONLY with 'yes' or 'no'."
        )
        try:
            res = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
            is_relevant = "yes" in res
        except Exception:
            is_relevant = True  # Safe fallback if grading call fails

        if is_relevant:
            relevant_documents.append(doc)

    log_msg = f"Document Grading: {len(relevant_documents)} / {len(documents)} chunks graded as relevant"
    trace.append(log_msg)

    return {
        "relevant_documents": relevant_documents,
        "trace": trace
    }

def node_increment_retry(state: GraphState) -> Dict[str, Any]:
    """
    Helper Node: Increments retry counter when no relevant chunks were found.
    """
    retry_count = state.get("retry_count", 0) + 1
    trace = list(state.get("trace", []))
    trace.append(f"Self-Correction: No relevant chunks found. Initiating retry #{retry_count} (limit: {state.get('max_retries', 2)})")
    return {
        "retry_count": retry_count,
        "trace": trace
    }

def node_generation(state: GraphState) -> Dict[str, Any]:
    """
    Node 4: Generation
    Synthesizes the final answer grounded in relevant documents with source citations.
    """
    question = state["question"]
    relevant_docs = state.get("relevant_documents", [])
    trace = list(state.get("trace", []))
    llm = get_llm()

    context_blocks = []
    sources = set()
    for doc in relevant_docs:
        src = doc.get("source", "documentation")
        sources.add(src)
        context_blocks.append(f"[Source: {src}]\n{doc['content']}")

    context_str = "\n\n---\n\n".join(context_blocks)

    prompt = (
        "You are an expert technical documentation assistant. Answer the user's question accurately "
        "and concisely using ONLY the provided technical context.\n"
        "Guidelines:\n"
        "- Ground your answer strictly in the context.\n"
        "- Include clear code examples if present in the context.\n"
        "- At the end of your response, list the cited sources under a '### Sources' header.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception as e:
        response = f"Failed to generate answer: {str(e)}"

    sources_list = sorted(list(sources))
    log_msg = f"Generation: generated answer citing {sources_list}"
    trace.append(log_msg)

    return {
        "answer": response,
        "sources": sources_list,
        "trace": trace
    }

def node_fallback(state: GraphState) -> Dict[str, Any]:
    """
    Fallback Node: Executed when max retries are exceeded without finding relevant docs.
    """
    trace = list(state.get("trace", []))
    log_msg = "Fallback: Query exceeded retry limit without finding relevant documentation."
    trace.append(log_msg)

    fallback_answer = (
        "I don't know based on the provided technical documentation. "
        "The retrieved documents did not contain sufficient relevant information to answer your question."
    )

    return {
        "answer": fallback_answer,
        "sources": [],
        "trace": trace
    }

# =====================================================================
# 3. Conditional Routing Logic
# =====================================================================

def decide_to_generate(state: GraphState) -> str:
    """
    Evaluates document grading outcome:
    - If relevant documents found -> proceed to 'generation'
    - If no relevant documents found:
      - if retry_count < max_retries -> proceed to 'increment_retry' and rewrite
      - if retry_count >= max_retries -> proceed to 'fallback'
    """
    relevant_docs = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if len(relevant_docs) > 0:
        return "generate"
    elif retry_count < max_retries:
        return "rewrite"
    else:
        return "fallback"

# =====================================================================
# 4. Build and Compile the StateGraph
# =====================================================================

def build_workflow():
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("query_analysis", node_query_analysis)
    workflow.add_node("retrieval", node_retrieval)
    workflow.add_node("document_grader", node_document_grader)
    workflow.add_node("increment_retry", node_increment_retry)
    workflow.add_node("generation", node_generation)
    workflow.add_node("fallback", node_fallback)

    # Define linear edges
    workflow.set_entry_point("query_analysis")
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "document_grader")

    # Define conditional routing from document_grader
    workflow.add_conditional_edges(
        "document_grader",
        decide_to_generate,
        {
            "generate": "generation",
            "rewrite": "increment_retry",
            "fallback": "fallback"
        }
    )

    # Loop back from retry increment to query analysis
    workflow.add_edge("increment_retry", "query_analysis")

    # End states
    workflow.add_edge("generation", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()

# Global app instance for execution
app_graph = build_workflow()

def run_rag_pipeline(question: str, max_retries: int = 2) -> Dict[str, Any]:
    """Helper function to execute the full LangGraph pipeline for a question."""
    initial_state: GraphState = {
        "question": question,
        "query_type": "how-to",
        "rewritten_query": question,
        "documents": [],
        "relevant_documents": [],
        "retry_count": 0,
        "max_retries": max_retries,
        "answer": "",
        "sources": [],
        "trace": []
    }
    final_state = app_graph.invoke(initial_state)
    return {
        "question": final_state["question"],
        "answer": final_state["answer"],
        "sources": final_state.get("sources", []),
        "retries": final_state.get("retry_count", 0),
        "query_type": final_state.get("query_type", "unknown"),
        "trace": final_state.get("trace", [])
    }

