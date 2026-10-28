from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from app.core.config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    validate_config,
)
from app.ingestion.loader import load_pdfs
from app.ingestion.splitter import split_documents
from app.rag.embeddings import get_embeddings


def index_pdfs():
    validate_config()

    # Load PDF pages
    documents = load_pdfs()

    # Split pages into chunks
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were generated.")

    # Connect to Qdrant
    client = QdrantClient(url=QDRANT_URL)

    try:
        if client.collection_exists(QDRANT_COLLECTION):
            client.delete_collection(QDRANT_COLLECTION)

        # Embed and store chunks
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION,
        )
    finally:
        client.close()

    print(f"Successfully indexed {len(chunks)} chunks into '{QDRANT_COLLECTION}'.")


if __name__ == "__main__":
    index_pdfs()
