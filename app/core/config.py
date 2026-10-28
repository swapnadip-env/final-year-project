import os
from pathlib import Path

from dotenv import load_dotenv

# Project root: final-year-project/
BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")

PDF_DIR = BASE_DIR / "app" / "data" 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)
QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "pdf_knowledge_base",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))


def validate_config():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Check your .env file.")

    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be positive.")

    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE.")
