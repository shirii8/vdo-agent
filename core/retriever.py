"""
Hybrid retrieval for the chat: keyword search + vector search, fused.

Why both:
  - Vector search matches meaning ("how do they log in" finds "authentication")
    but can rank a vague chunk above the one that literally contains the
    asked-for phrase.
  - BM25 keyword search matches exact words ("identity protocols", "SAML")
    but misses paraphrases.
The two ranked lists are merged with Reciprocal Rank Fusion (RRF): a chunk
scores 1 / (60 + rank) in each list it appears in, and the sums are sorted.

Transcript chunks are short (~30 s of speech), so an explanation is usually
spread over consecutive chunks. After picking the best chunks, their
neighbours are added too, and everything is returned in video order so the
model reads continuous passages.
"""

import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

RRF_K = 60        # standard RRF constant; damps the weight of top ranks
CANDIDATES = 12   # how many results to take from each search before fusing


def _tokens(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self, vector_store, docs: list, k: int = 4, neighbors: int = 1):
        self.vector_store = vector_store
        # Chunks in video order; position in this list == metadata["chunk_index"].
        self.docs = sorted(docs, key=lambda d: d.metadata["chunk_index"])
        self.k = k
        self.neighbors = neighbors
        self.bm25 = BM25Okapi([_tokens(d.page_content) for d in self.docs])

    def rank(self, question: str, k: int | None = None) -> list:
        """Top-k chunk indices for the question, best first (no neighbours)."""
        k = k or self.k
        n = min(CANDIDATES, len(self.docs))

        vector_hits = [
            d.metadata["chunk_index"] for d in self.vector_store.similarity_search(question, k=n)
        ]

        scores = self.bm25.get_scores(_tokens(question))
        keyword_hits = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[:n]
        # Drop chunks that share no word with the question.
        keyword_hits = [i for i in keyword_hits if scores[i] > 0]

        fused = {}
        for hits in (vector_hits, keyword_hits):
            for rank, index in enumerate(hits):
                fused[index] = fused.get(index, 0.0) + 1.0 / (RRF_K + rank + 1)

        return sorted(fused, key=fused.get, reverse=True)[:k]

    def search(self, question: str) -> list:
        """Best chunks plus their neighbours, as Documents in video order."""
        wanted = set()
        for index in self.rank(question):
            for j in range(index - self.neighbors, index + self.neighbors + 1):
                if 0 <= j < len(self.docs):
                    wanted.add(j)
        return [self.docs[i] for i in sorted(wanted)]


def docs_from_store(vector_store) -> list:
    """Rebuild the chunk Documents from an existing Chroma collection."""
    data = vector_store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
