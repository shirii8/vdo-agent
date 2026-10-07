"""
ChromaDB vector store for the transcript.

Timestamped transcript segments are grouped into ~500-char chunks, each chunk
is embedded with a local HuggingFace sentence-transformer, and the vectors are
saved to ./vector_db so the RAG engine can find the chunks most similar to a
question. Every chunk keeps the start/end time (seconds) it covers, which the
chat uses to cite and link back to that moment in the video.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"  # on-disk folder for the Chroma database
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, runs on CPU (384-dim vectors)
CHUNK_CHARS = 500  # target chunk size: small chunks give precise retrieval


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def segments_to_documents(segments: list, chunk_chars: int = CHUNK_CHARS) -> list:
    """
    Group consecutive segments into ~chunk_chars Documents with start/end times.
    Chunks never split a segment, and each chunk repeats the previous chunk's
    last segment (overlap) so a sentence on the boundary appears in both.
    """
    docs, current = [], []

    def flush():
        docs.append(
            Document(
                page_content=" ".join(s["text"] for s in current),
                metadata={
                    "chunk_index": len(docs),
                    "start": float(current[0]["start"]),
                    "end": float(current[-1]["end"]),
                },
            )
        )

    for seg in segments:
        current.append(seg)
        if sum(len(s["text"]) + 1 for s in current) >= chunk_chars:
            flush()
            current = [current[-1]] if len(current) > 1 else []

    # Leftover segments (skip if it's only the overlap copy of the last chunk).
    if current and (not docs or current[-1]["end"] > docs[-1].metadata["end"]):
        flush()

    return docs


def build_vector_store(segments: list) -> Chroma:
    print("Building vector Store")

    docs = segments_to_documents(segments)

    embeddings = get_embeddings()

    # The collection persists on disk, so wipe the previous meeting first;
    # otherwise answers would mix chunks from every video ever analysed.
    Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    ).delete_collection()

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    return vector_store


def load_vector_store() -> Chroma:
    """Open the existing collection in vector_db/ without re-embedding."""
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    # Return the k chunks whose vectors are closest to the query's vector.
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
