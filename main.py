import os
import json
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import get_chroma_collection
from ingest import ingest_text, ingest_url, ingest_all_documents
from graph import run_rag_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure documents are indexed
    try:
        collection = get_chroma_collection()
        if collection.count() == 0:
            print("[Startup] Collection is empty. Ingesting default document corpus...")
            ingest_all_documents()
        else:
            print(f"[Startup] Found {collection.count()} existing document chunks in ChromaDB.")
    except Exception as e:
        print(f"[Startup Warning] Could not check collection: {e}")
    yield
    # Shutdown logic (if any)

# Initialize FastAPI application
app = FastAPI(
    title="Technical Documentation Assistant API",
    description="A Self-Corrective RAG system built with LangGraph, ChromaDB, and FastAPI",
    version="1.0.0"
)

# Enable CORS for local testing and frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "feedback.json")

# =====================================================================
# Pydantic Request & Response Schemas
# =====================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., example="How do I validate request bodies in FastAPI?")
    max_retries: Optional[int] = Field(default=2, ge=0, le=5, description="Maximum self-corrective retry attempts")

class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    sources: List[str]
    retries: int
    query_type: str
    trace: List[str]

class IngestTextRequest(BaseModel):
    title: str = Field(..., example="custom_guide.md")
    content: str = Field(..., example="This is the content of the guide...")

class IngestUrlRequest(BaseModel):
    url: str = Field(..., example="https://example.com/docs")

class FeedbackRequest(BaseModel):
    query_id: Optional[str] = Field(default=None, example="123e4567-e89b-12d3-a456-426614174000")
    question: Optional[str] = Field(default=None, example="How do I use Depends in FastAPI?")
    rating: str = Field(..., example="thumbs_up", description="thumbs_up or thumbs_down")
    comment: Optional[str] = Field(default=None, example="Very clear explanation and code sample.")

# =====================================================================
# API Endpoints
# =====================================================================

@app.post("/query", response_model=QueryResponse, summary="Submit a question to the assistant")
def query_assistant(request: QueryRequest):
    """
    Submits a user question to the LangGraph RAG pipeline.
    Executes Query Analysis -> Retrieval -> Document Grading -> (Retry loop if needed) -> Generation.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    try:
        result = run_rag_pipeline(request.question, max_retries=request.max_retries or 2)
        query_id = str(uuid.uuid4())
        return QueryResponse(
            query_id=query_id,
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            retries=result["retries"],
            query_type=result["query_type"],
            trace=result["trace"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(e)}"
        )

@app.get("/documents", summary="List all indexed documents in the vector database")
def list_documents():
    """Returns a list of all indexed documents and their chunk counts."""
    collection = get_chroma_collection()
    data = collection.get(include=["metadatas"])
    
    docs_summary = {}
    if data and data.get("metadatas"):
        for meta in data["metadatas"]:
            if meta and "source" in meta:
                src = meta["source"]
                docs_summary[src] = docs_summary.get(src, 0) + 1

    formatted_docs = [
        {"title": title, "chunk_count": count}
        for title, count in sorted(docs_summary.items())
    ]

    return {
        "total_chunks": collection.count(),
        "total_documents": len(formatted_docs),
        "documents": formatted_docs
    }

@app.post("/ingest", summary="Ingest new documents via file upload, JSON content, or URL")
def ingest_document(
    file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    url: Optional[str] = Form(None)
):
    """
    Ingests documents into ChromaDB.
    Supports:
    1. File upload (.md or .txt)
    2. URL (fetches web page text)
    3. Direct title & text content
    """
    try:
        # Case 1: Uploaded file
        if file:
            file_bytes = file.file.read()
            text_content = file_bytes.decode("utf-8", errors="replace")
            doc_title = file.filename or "uploaded_file.txt"
            chunks = ingest_text(title=doc_title, content=text_content)
            return {
                "status": "success",
                "message": f"Successfully ingested {chunks} chunks from file '{doc_title}'.",
                "chunks_ingested": chunks
            }

        # Case 2: URL
        if url:
            chunks = ingest_url(url.strip())
            return {
                "status": "success",
                "message": f"Successfully fetched and ingested {chunks} chunks from '{url}'.",
                "chunks_ingested": chunks
            }

        # Case 3: Direct title and content
        if content and title:
            chunks = ingest_text(title=title.strip(), content=content.strip())
            return {
                "status": "success",
                "message": f"Successfully ingested {chunks} chunks for '{title}'.",
                "chunks_ingested": chunks
            }

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide a file, a URL, or title and content to ingest."
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )

@app.post("/feedback", summary="Submit feedback on an answer")
def submit_feedback(request: FeedbackRequest):
    """Stores user feedback (thumbs up / thumbs down + comments) in feedback.json."""
    if request.rating not in ["thumbs_up", "thumbs_down"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rating must be either 'thumbs_up' or 'thumbs_down'."
        )

    feedback_entry = {
        "feedback_id": str(uuid.uuid4()),
        "query_id": request.query_id,
        "question": request.question,
        "rating": request.rating,
        "comment": request.comment,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Load existing feedback or create empty list
    feedbacks = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
        except Exception:
            feedbacks = []

    feedbacks.append(feedback_entry)

    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedbacks, f, indent=2)

    return {
        "status": "success",
        "message": "Feedback submitted successfully.",
        "feedback": feedback_entry
    }

@app.get("/feedback", summary="View all submitted feedback")
def get_all_feedback():
    """Lists all stored feedback entries."""
    if not os.path.exists(FEEDBACK_FILE):
        return {"total_feedback": 0, "feedback": []}
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            feedbacks = json.load(f)
        return {"total_feedback": len(feedbacks), "feedback": feedbacks}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read feedback: {str(e)}"
        )

# =====================================================================
# Interactive Web UI (Bonus: Clean single-page browser frontend)
# =====================================================================

@app.get("/", response_class=HTMLResponse, summary="Interactive Web Frontend")
def serve_ui():
    """Serves a clean, beginner-friendly web UI to test Q&A, documents, and feedback."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAG Technical Documentation Assistant</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    pre { white-space: pre-wrap; word-break: break-word; }
  </style>
</head>
<body class="bg-slate-50 min-h-screen text-slate-800">
  <!-- Header -->
  <header class="bg-indigo-700 text-white shadow-md">
    <div class="max-w-5xl mx-auto px-4 py-4 flex justify-between items-center">
      <div>
        <h1 class="text-xl font-bold flex items-center gap-2">
          <i class="fa-solid fa-brain"></i> RAG Technical Assistant
        </h1>
        <p class="text-xs text-indigo-200">Self-Corrective LangGraph Workflow &bull; FastAPI Backend</p>
      </div>
      <div class="text-xs space-x-2">
        <a href="/docs" target="_blank" class="bg-indigo-600 hover:bg-indigo-800 px-3 py-1.5 rounded transition font-medium">
          <i class="fa-solid fa-book-open"></i> API Docs (/docs)
        </a>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-5xl mx-auto px-4 py-8 grid grid-cols-1 md:grid-cols-3 gap-6">
    
    <!-- Left Column: Q&A Section -->
    <div class="md:col-span-2 space-y-6">
      
      <!-- Ask Question Card -->
      <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
        <h2 class="text-lg font-semibold text-slate-800 mb-3 flex items-center gap-2">
          <i class="fa-solid fa-comment-dots text-indigo-600"></i> Ask Documentation
        </h2>
        
        <!-- Sample Questions -->
        <div class="mb-3">
          <p class="text-xs text-slate-500 font-medium mb-1.5">Try a sample query:</p>
          <div class="flex flex-wrap gap-2">
            <button onclick="setQuery('How to validate request body with Pydantic in FastAPI?')" class="text-xs bg-slate-100 hover:bg-indigo-50 hover:text-indigo-600 px-2.5 py-1 rounded-full border border-slate-200 transition">
              Pydantic validation
            </button>
            <button onclick="setQuery('How does Depends work for dependency injection?')" class="text-xs bg-slate-100 hover:bg-indigo-50 hover:text-indigo-600 px-2.5 py-1 rounded-full border border-slate-200 transition">
              Dependency injection
            </button>
            <button onclick="setQuery('How to configure CORS middleware in FastAPI?')" class="text-xs bg-slate-100 hover:bg-indigo-50 hover:text-indigo-600 px-2.5 py-1 rounded-full border border-slate-200 transition">
              CORS middleware
            </button>
            <button onclick="setQuery('How to train an XGBoost machine learning model?')" class="text-xs bg-amber-50 hover:bg-amber-100 text-amber-700 px-2.5 py-1 rounded-full border border-amber-200 transition" title="Tests Fallback logic">
              Out-of-domain (Test Fallback)
            </button>
          </div>
        </div>

        <form id="queryForm" onsubmit="handleQuery(event)" class="space-y-3">
          <textarea id="questionInput" rows="3" required placeholder="Type your technical question here..." class="w-full border border-slate-300 rounded-lg p-3 text-sm focus:ring-2 focus:ring-indigo-500 focus:outline-none"></textarea>
          
          <div class="flex justify-between items-center">
            <div class="flex items-center gap-2 text-xs text-slate-500">
              <label for="maxRetries">Max Retries:</label>
              <input type="number" id="maxRetries" value="2" min="0" max="4" class="w-14 border border-slate-300 rounded px-2 py-1 text-center">
            </div>
            <button type="submit" id="submitBtn" class="bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium px-5 py-2 rounded-lg transition flex items-center gap-2">
              <i class="fa-solid fa-paper-plane"></i> Ask Assistant
            </button>
          </div>
        </form>
      </div>

      <!-- Loading Indicator -->
      <div id="loadingIndicator" class="hidden bg-white p-6 rounded-xl shadow-sm border border-slate-200 text-center space-y-3">
        <i class="fa-solid fa-spinner fa-spin text-3xl text-indigo-600"></i>
        <p class="text-sm text-slate-600 font-medium">Running LangGraph self-corrective workflow...</p>
        <p class="text-xs text-slate-400">Query Analysis &rarr; ChromaDB Retrieval &rarr; Document Grading &rarr; Generation</p>
      </div>

      <!-- Answer Card -->
      <div id="answerCard" class="hidden bg-white p-6 rounded-xl shadow-sm border border-slate-200 space-y-4">
        <div class="flex justify-between items-center border-b border-slate-100 pb-3">
          <span class="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <i class="fa-solid fa-robot text-indigo-600"></i> Assistant Response
          </span>
          <div id="metricsBadges" class="flex items-center gap-2 text-xs"></div>
        </div>

        <!-- Answer Content -->
        <div id="answerText" class="text-sm text-slate-800 leading-relaxed space-y-2 whitespace-pre-wrap"></div>

        <!-- Sources -->
        <div id="sourcesContainer" class="pt-3 border-t border-slate-100">
          <p class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Sources Cited:</p>
          <div id="sourcesBadges" class="flex flex-wrap gap-2"></div>
        </div>

        <!-- Workflow Execution Trace Accordion -->
        <details class="bg-slate-50 p-3 rounded-lg border border-slate-200 text-xs">
          <summary class="font-medium text-slate-600 cursor-pointer hover:text-indigo-600">
            <i class="fa-solid fa-code-branch"></i> View LangGraph Node Execution Trace
          </summary>
          <ul id="traceList" class="mt-2 space-y-1 text-slate-600 pl-4 list-disc"></ul>
        </details>

        <!-- Feedback Section -->
        <div id="feedbackSection" class="pt-3 border-t border-slate-100 flex items-center justify-between">
          <p class="text-xs text-slate-500">Was this answer helpful?</p>
          <div class="flex items-center gap-2">
            <button onclick="sendFeedback('thumbs_up')" class="hover:bg-slate-100 text-slate-600 hover:text-green-600 p-2 rounded text-sm transition">
              <i class="fa-regular fa-thumbs-up"></i>
            </button>
            <button onclick="sendFeedback('thumbs_down')" class="hover:bg-slate-100 text-slate-600 hover:text-red-600 p-2 rounded text-sm transition">
              <i class="fa-regular fa-thumbs-down"></i>
            </button>
          </div>
        </div>
        <div id="feedbackNotice" class="hidden text-xs text-green-600 font-medium text-right"></div>
      </div>

    </div>

    <!-- Right Column: Corpus & Ingestion -->
    <div class="space-y-6">
      
      <!-- Documents in Corpus -->
      <div class="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
        <div class="flex justify-between items-center mb-3">
          <h3 class="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <i class="fa-solid fa-folder-open text-indigo-600"></i> Indexed Corpus
          </h3>
          <button onclick="loadDocuments()" class="text-xs text-indigo-600 hover:underline">
            <i class="fa-solid fa-arrows-rotate"></i> Refresh
          </button>
        </div>
        <div id="documentsList" class="text-xs text-slate-600 space-y-2">
          <p class="text-slate-400">Loading documents...</p>
        </div>
      </div>

      <!-- Quick Ingest Card -->
      <div class="bg-white p-5 rounded-xl shadow-sm border border-slate-200 space-y-3">
        <h3 class="text-sm font-semibold text-slate-700 flex items-center gap-2">
          <i class="fa-solid fa-file-arrow-up text-indigo-600"></i> Ingest Document
        </h3>
        <p class="text-xs text-slate-500">Upload a new text/markdown file to index into ChromaDB:</p>
        
        <form id="uploadForm" onsubmit="handleUpload(event)" class="space-y-2">
          <input type="file" id="fileInput" accept=".txt,.md" class="text-xs w-full border border-slate-300 rounded p-1.5">
          <button type="submit" class="w-full bg-slate-800 hover:bg-slate-900 text-white text-xs py-2 rounded transition font-medium">
            Upload & Ingest
          </button>
        </form>
        <p id="uploadNotice" class="hidden text-xs font-medium"></p>
      </div>

    </div>

  </main>

  <!-- JavaScript logic -->
  <script>
    let currentQueryId = null;
    let currentQuestion = null;

    function setQuery(text) {
      document.getElementById('questionInput').value = text;
    }

    async function loadDocuments() {
      const container = document.getElementById('documentsList');
      try {
        const res = await fetch('/documents');
        const data = await res.json();
        if (data.documents && data.documents.length > 0) {
          container.innerHTML = `
            <div class="mb-2 font-medium text-slate-700">Total Chunks: <span class="text-indigo-600 font-bold">${data.total_chunks}</span></div>
            <ul class="space-y-1.5 border-t border-slate-100 pt-2">
              ${data.documents.map(d => `
                <li class="flex justify-between items-center p-1.5 bg-slate-50 rounded">
                  <span class="truncate font-mono">${d.title}</span>
                  <span class="bg-indigo-100 text-indigo-700 px-1.5 py-0.5 rounded text-[10px]">${d.chunk_count} chunks</span>
                </li>
              `).join('')}
            </ul>
          `;
        } else {
          container.innerHTML = '<p class="text-slate-400">No documents found.</p>';
        }
      } catch (err) {
        container.innerHTML = '<p class="text-red-500">Error loading documents.</p>';
      }
    }

    async function handleQuery(e) {
      e.preventDefault();
      const question = document.getElementById('questionInput').value.trim();
      const maxRetries = parseInt(document.getElementById('maxRetries').value) || 2;
      if (!question) return;

      currentQuestion = question;
      document.getElementById('submitBtn').disabled = true;
      document.getElementById('loadingIndicator').classList.remove('hidden');
      document.getElementById('answerCard').classList.add('hidden');
      document.getElementById('feedbackNotice').classList.add('hidden');

      try {
        const res = await fetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, max_retries: maxRetries })
        });
        const data = await res.json();

        currentQueryId = data.query_id;

        // Render answer
        document.getElementById('answerText').textContent = data.answer;

        // Metrics badges
        const metrics = document.getElementById('metricsBadges');
        metrics.innerHTML = `
          <span class="bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded font-mono text-xs">Type: ${data.query_type}</span>
          <span class="bg-amber-100 text-amber-700 px-2 py-0.5 rounded font-mono text-xs">Retries: ${data.retries}</span>
        `;

        // Sources
        const sourcesDiv = document.getElementById('sourcesBadges');
        if (data.sources && data.sources.length > 0) {
          sourcesDiv.innerHTML = data.sources.map(s => `
            <span class="bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-1 rounded text-xs flex items-center gap-1 font-mono">
              <i class="fa-regular fa-file-code"></i> ${s}
            </span>
          `).join('');
        } else {
          sourcesDiv.innerHTML = '<span class="text-xs text-slate-400">None (Fallback / General)</span>';
        }

        // Trace
        const traceList = document.getElementById('traceList');
        traceList.innerHTML = (data.trace || []).map(t => `<li class="font-mono text-[11px]">${t}</li>`).join('');

        document.getElementById('answerCard').classList.remove('hidden');
      } catch (err) {
        alert('Error contacting server: ' + err.message);
      } finally {
        document.getElementById('submitBtn').disabled = false;
        document.getElementById('loadingIndicator').classList.add('hidden');
      }
    }

    async function sendFeedback(rating) {
      if (!currentQueryId) return;
      try {
        const res = await fetch('/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query_id: currentQueryId,
            question: currentQuestion,
            rating: rating,
            comment: rating === 'thumbs_up' ? 'Helpful answer' : 'Needs improvement'
          })
        });
        const data = await res.json();
        const notice = document.getElementById('feedbackNotice');
        notice.textContent = 'Feedback recorded. Thank you!';
        notice.classList.remove('hidden');
      } catch (err) {
        console.error(err);
      }
    }

    async function handleUpload(e) {
      e.preventDefault();
      const fileInput = document.getElementById('fileInput');
      if (!fileInput.files.length) return;

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);

      const notice = document.getElementById('uploadNotice');
      notice.textContent = 'Uploading and chunking...';
      notice.className = 'text-xs text-indigo-600 font-medium';
      notice.classList.remove('hidden');

      try {
        const res = await fetch('/ingest', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        notice.textContent = data.message;
        notice.className = 'text-xs text-emerald-600 font-medium';
        fileInput.value = '';
        loadDocuments();
      } catch (err) {
        notice.textContent = 'Upload failed: ' + err.message;
        notice.className = 'text-xs text-red-600 font-medium';
      }
    }

    // Initial load
    loadDocuments();
  </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
