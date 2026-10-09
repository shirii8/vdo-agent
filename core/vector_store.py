"""
ChromaDB vector store for the transcript.

Timestamped transcript segments are grouped into ~500-char chunks, each chunk
is embedded with a local HuggingFace sentence-transformer, and the vectors go
into an in-memory Chroma collection so the RAG engine can find the chunks most
similar to a question. Every chunk keeps the start/end time (seconds) it
covers, which the chat uses to cite and link back to that moment in the video.

Each analysis gets its own collection with a unique name. Nothing is shared or
written to disk, so two people (or two browser tabs) analysing at the same
time cannot overwrite or corrupt each other's index.
"""

import threading
import uuid

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, runs on CPU (384-dim vectors)
CHUNK_CHARS = 500  # target chunk size: small chunks give precise retrieval

# One Chroma client and one embedding model for the whole process, created the
# first time they are needed. The lock matters: Chroma is not safe when two
# threads create a client at the same moment (one sees it half-built and fails
# with "'RustBindingsAPI' object has no attribute 'bindings'").
_lock = threading.Lock()
_client = None
_embeddings = None


def get_client():
    global _client
    with _lock:
        if _client is None:
            _client = chromadb.EphemeralClient()  # in memory, nothing on disk
        return _client


def get_embeddings():
    global _embeddings
    with _lock:
        if _embeddings is None:
            # Loading the model takes several seconds, so do it once per process.
            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
            )
        return _embeddings


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

    # A fresh collection on the shared in-memory client, under a name no other
    # analysis uses.
    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        client=get_client(),
        collection_name=f"transcript_{uuid.uuid4().hex}",
    )
