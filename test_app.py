"""
Fast and reliable test suite verifying all assignment endpoints directly.
Tests:
1. Document Listing (/documents)
2. LangGraph Query Pipeline (/query)
3. Feedback Submission (/feedback)
4. Feedback Retrieval (/feedback)
5. Ingestion of new content (/ingest)
6. Web UI HTML rendering (/)
"""
from main import (
    list_documents,
    query_assistant,
    submit_feedback,
    get_all_feedback,
    ingest_document,
    serve_ui,
    QueryRequest,
    FeedbackRequest
)

def test_suite():
    print("=" * 60)
    print("RUNNING RAG TECHNICAL ASSISTANT TEST SUITE")
    print("=" * 60)

    # 1. Test GET /documents
    print("\n[TEST 1] Testing list_documents() (GET /documents)...")
    docs_result = list_documents()
    print(f"Total Chunks: {docs_result['total_chunks']}")
    print(f"Total Documents: {docs_result['total_documents']}")
    for d in docs_result['documents']:
        print(f"  - {d['title']} ({d['chunk_count']} chunks)")
    assert docs_result["total_chunks"] > 0, "No chunks found in database!"
    assert docs_result["total_documents"] >= 5, "Expected at least 5 documents!"
    print("-> PASS: /documents endpoint verified.")

    # 2. Test POST /query
    print("\n[TEST 2] Testing query_assistant() (POST /query)...")
    req = QueryRequest(
        question="How do I validate request bodies using Pydantic in FastAPI?",
        max_retries=2
    )
    query_result = query_assistant(req)
    print(f"Query ID: {query_result.query_id}")
    print(f"Query Type: {query_result.query_type}")
    print(f"Retries: {query_result.retries}")
    print(f"Sources: {query_result.sources}")
    print(f"Answer Preview: {query_result.answer[:120]}...")
    assert query_result.answer, "Answer should not be empty!"
    assert len(query_result.trace) > 0, "Workflow trace should not be empty!"
    print("-> PASS: /query endpoint & LangGraph workflow verified.")

    # 3. Test POST /feedback
    print("\n[TEST 3] Testing submit_feedback() (POST /feedback)...")
    fb_req = FeedbackRequest(
        query_id=query_result.query_id,
        question=req.question,
        rating="thumbs_up",
        comment="Accurate explanation with code example."
    )
    fb_result = submit_feedback(fb_req)
    print(f"Feedback Status: {fb_result['status']}")
    assert fb_result["status"] == "success", "Feedback submission failed!"
    print("-> PASS: POST /feedback verified.")

    # 4. Test GET /feedback
    print("\n[TEST 4] Testing get_all_feedback() (GET /feedback)...")
    all_fb = get_all_feedback()
    print(f"Total Feedbacks: {all_fb['total_feedback']}")
    assert all_fb["total_feedback"] >= 1, "Expected at least 1 feedback record!"
    print("-> PASS: GET /feedback verified.")

    # 5. Test POST /ingest
    print("\n[TEST 5] Testing ingest_document() (POST /ingest)...")
    ingest_result = ingest_document(
        file=None,
        title="test_quickstart.md",
        content="FastAPI is an asynchronous web framework built on Starlette and Pydantic.",
        url=None
    )
    print(f"Ingest Message: {ingest_result['message']}")
    assert ingest_result["status"] == "success", "Document ingestion failed!"
    print("-> PASS: /ingest endpoint verified.")

    # 6. Test GET /
    print("\n[TEST 6] Testing serve_ui() (GET /)...")
    ui_html = serve_ui()
    assert "RAG Technical Assistant" in ui_html
    assert "Ask Documentation" in ui_html
    print("-> PASS: Web UI HTML rendering verified.")

    print("\n" + "=" * 60)
    print("ALL 6 VERIFICATION TESTS PASSED SUCCESSFULLY! [SUCCESS]")
    print("=" * 60)

if __name__ == "__main__":
    test_suite()
