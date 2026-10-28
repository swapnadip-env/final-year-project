
from langchain_community.document_loaders import PyPDFLoader

from app.core.config import PDF_DIR


def load_pdfs():
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIR}"
        )

    documents = []

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            # Preserve useful source metadata.
            page.metadata["source"] = pdf_path.name

        documents.extend(pages)

    print(
        f"Loaded {len(documents)} pages "
        f"from {len(pdf_files)} PDF file(s)."
    )

    return documents
