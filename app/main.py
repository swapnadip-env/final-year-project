
from app.core.config import validate_config
from app.rag.chain import answer_question


def main():
    validate_config()

    print("=" * 55)
    print("PDF QUESTION-ANSWERING RAG APP")
    print("=" * 55)
    print("Ask questions about your indexed PDFs.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = answer_question(question)

            print("\nAssistant:")
            print(result["answer"])

            print("\nSources:")

            if result["sources"]:
                for source in result["sources"]:
                    print(
                        f"- {source['source']}, "
                        f"page {source['page'] or 'unknown'}"
                    )
            else:
                print("- No sources found.")

            print()

        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
