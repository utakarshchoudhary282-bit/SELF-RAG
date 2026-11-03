import os
import glob
import re
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import get_chroma_collection

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")

# Chunking strategy parameters:
# - chunk_size=500: Keeps technical concepts and code snippets concise and focused.
# - chunk_overlap=50: Prevents loss of context across split boundaries.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

def chunk_text(text: str, source: str):
    """Splits a single document text into chunks with source metadata."""
    chunks = text_splitter.split_text(text)
    records = []
    for idx, chunk in enumerate(chunks):
        records.append({
            "id": f"{source}_chunk_{idx}",
            "text": chunk,
            "metadata": {
                "source": source,
                "chunk_index": idx
            }
        })
    return records

def ingest_text(title: str, content: str):
    """Indexes raw text under a given title."""
    collection = get_chroma_collection()
    records = chunk_text(content, title)
    if not records:
        return 0

    ids = [r["id"] for r in records]
    documents = [r["text"] for r in records]
    metadatas = [r["metadata"] for r in records]

    # Upsert into ChromaDB
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"[Ingest] Ingested {len(records)} chunks for '{title}'.")
    return len(records)

def ingest_url(url: str):
    """Fetches text from a URL and indexes it."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    # Strip HTML tags simply if HTML
    clean_text = re.sub(r"<[^>]+>", " ", response.text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    return ingest_text(title=url, content=clean_text)

def ingest_all_documents():
    """Reads all markdown files from data/documents/ and indexes them in ChromaDB."""
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    files = glob.glob(os.path.join(DOCUMENTS_DIR, "*.md")) + glob.glob(os.path.join(DOCUMENTS_DIR, "*.txt"))

    if not files:
        print(f"[Ingest] No files found in {DOCUMENTS_DIR}")
        return 0

    total_chunks = 0
    for file_path in files:
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        num_chunks = ingest_text(title=file_name, content=content)
        total_chunks += num_chunks

    print(f"\n[Ingest Complete] Indexed {len(files)} files into {total_chunks} total chunks.")
    return total_chunks

if __name__ == "__main__":
    print("[Ingest] Starting document indexing...")
    ingest_all_documents()

