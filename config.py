import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()

# Vector database storage path
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "tech_docs"

# Singleton ChromaDB client and collection to avoid SQLite lock contention
_chroma_client = None
_chroma_collection = None

def get_chroma_collection():
    """Returns the persistent ChromaDB collection with local embeddings (Singleton)."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        # DefaultEmbeddingFunction uses local ONNX all-MiniLM-L6-v2 (runs offline on CPU, free)
        embedding_func = embedding_functions.DefaultEmbeddingFunction()
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func
        )
    return _chroma_collection

def get_llm():
    """
    Returns an LLM instance based on available API keys in .env.
    Prioritizes OpenAI (gpt-4o-mini), Groq, or Google Gemini.
    """
    provider = os.getenv("LLM_PROVIDER", "").lower()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    # Priority 1: OpenAI (gpt-4o-mini recommended by the assignment)
    if (provider == "openai" or not provider) and openai_api_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.0
        )
    # Priority 2: Groq
    elif (provider == "groq" or not provider) and groq_api_key:
        from langchain_groq import ChatGroq
        # Try available Groq chat model
        return ChatGroq(
            model_name="openai/gpt-oss-20b",
            groq_api_key=groq_api_key,
            temperature=0.0
        )
    # Priority 3: Google Gemini
    elif (provider == "gemini" or provider == "google" or not provider) and google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.0
        )
    else:
        # Fallback simulated Demo mode
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        return FakeListChatModel(responses=[
            "Rewritten Query for retrieval",
            "yes",
            "yes",
            "yes",
            "yes",
            "Based on the technical documentation, FastAPI provides automatic validation with Pydantic and easy dependency injection with Depends.\n\n### Sources\n- 02_pydantic_validation.md\n- 03_dependency_injection.md"
        ])
