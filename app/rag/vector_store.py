from langchain_qdrant import QdrantVectorStore

from app.core.config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
)
from app.rag.embeddings import get_embeddings


def get_vector_store():
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        collection_name=QDRANT_COLLECTION,
        url=QDRANT_URL,
    )
