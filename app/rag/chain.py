from langchain_groq import ChatGroq

from app.core.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    RETRIEVAL_K,
)
from app.rag.vector_store import get_vector_store
from app.rag.prompts import RAG_PROMPT


def build_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )


def answer_question(question: str) -> dict:
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    # Retrieve the vector store
    vector_store = get_vector_store()

    # Find the most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "answer": ("I could not find relevant information in the indexed PDFs."),
            "sources": [],
        }

    # Prepare context for the LLM
    context_parts = []

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page")

        page_label = str(page + 1) if isinstance(page, int) else "Unknown"

        context_parts.append(
            f"Source: {source}, Page: {page_label}\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Generate an answer using Groq
    prompt_value = RAG_PROMPT.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    response = build_llm().invoke(prompt_value)

    # Return the answer and source references
    sources = []

    for doc in retrieved_docs:
        source_info = {
            "source": doc.metadata.get("source", "Unknown PDF"),
            "page": (
                doc.metadata["page"] + 1
                if isinstance(doc.metadata.get("page"), int)
                else None
            ),
        }

        if source_info not in sources:
            sources.append(source_info)

    return {
        "answer": response.content,
        "sources": sources,
    }
