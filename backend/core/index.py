"""Build the vector index from data/sample_docs. Run via `make index`.

Indexing is separate from serving on purpose: embedding the corpus takes seconds
and only needs to happen when the documents change, so paying it inside a request
would be waste. Later techniques (Graph RAG's entity extraction, BM25's term
statistics) hook in here for the same reason.
"""

from core import embeddings, vectorstore
from core.chunking import split_text
from core.config import settings
from core.ingest import load_documents


def build_index() -> dict:
    """Re-index the sample corpus from scratch. Returns counts for the CLI to print."""
    documents = load_documents(settings.sample_docs_dir)
    if not documents:
        raise RuntimeError(f"No documents found in {settings.sample_docs_dir}")

    chunks: list[vectorstore.IndexedChunk] = []
    for document in documents:
        for position, text in enumerate(
            split_text(document.text, settings.chunk_size, settings.chunk_overlap)
        ):
            chunks.append(
                vectorstore.IndexedChunk(
                    # Positional id: stable across runs as long as the document is
                    # unchanged, and readable in the UI ("dynamo.md#3").
                    chunk_id=f"{document.source}#{position}",
                    text=text,
                    source=document.source,
                )
            )

    # Drop first: chunk ids are positional, so editing a document shortens the list
    # and would strand the tail chunks of the previous run in the collection.
    vectorstore.reset_collection()
    vectorstore.add_chunks(chunks, embeddings.embed_texts([c.text for c in chunks]))

    return {"documents": len(documents), "chunks": len(chunks)}


def main() -> None:
    result = build_index()
    print(f"Indexed {result['chunks']} chunks from {result['documents']} documents.")


if __name__ == "__main__":
    main()
