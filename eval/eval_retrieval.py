"""
Measures how good the RAG retrieval is on known videos.

Each answerable question has a hand-labelled "gold" time range: the part of the
video that contains the answer. A question counts as a hit at k if any of the
top-k retrieved chunks overlaps that range.

  recall@k = hits / answerable questions
  MRR      = mean of 1/rank of the first correct chunk (0 if not in top k)

Two retrieval modes are compared on the same chunks:
  vector - embedding similarity only (the original retriever)
  hybrid - BM25 keyword search + vector search, fused (core/retriever.py)

With --answers it also calls Gemini through the real chat chain (hybrid
retrieval + neighbouring chunks) and checks behaviour: answerable questions
must get an answer, and questions marked "answerable": false must get the
"could not find" reply.

Usage (from the project root):
    python eval/eval_retrieval.py              # retrieval only, no API calls
    python eval/eval_retrieval.py --answers    # + answer/abstain check (uses Gemini)
"""

import argparse
import json
import os
import sys

# Allow "from core..." when run as a script from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from core.vector_store import get_embeddings, segments_to_documents
from core.retriever import HybridRetriever
from core.rag_engine import _make_chain, format_timestamp, NOT_FOUND

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, cached transcript segments, questions)
DATASETS = [
    ("ai_economy", "ai_economy_segments.json", "questions.json"),
    ("authentication", "auth_segments.json", "auth_questions.json"),
]


def load(name: str):
    return json.load(open(os.path.join(HERE, name), encoding="utf-8"))


def first_hit_rank(indices: list, docs: list, gold_start: float, gold_end: float):
    """1-based rank of the first chunk whose time span overlaps the gold range."""
    for rank, index in enumerate(indices, start=1):
        meta = docs[index].metadata
        if meta["start"] <= gold_end and meta["end"] >= gold_start:
            return rank
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--answers", action="store_true", help="also check answers with Gemini")
    args = parser.parse_args()

    cutoffs = sorted({1, 3, args.k})
    totals = {mode: {"n": 0, "rr": 0.0, **{c: 0 for c in cutoffs}} for mode in ("vector", "hybrid")}
    behaviour_ok = behaviour_total = 0

    for name, segments_file, questions_file in DATASETS:
        segments = load(segments_file)["segments"]
        questions = load(questions_file)
        docs = segments_to_documents(segments)

        # A separate in-memory collection per dataset.
        store = Chroma.from_documents(docs, get_embeddings(), collection_name=f"eval_{name}")
        hybrid = HybridRetriever(store, docs, k=args.k)

        def vector_rank(question):
            return [d.metadata["chunk_index"] for d in store.similarity_search(question, k=args.k)]

        print(f"\n== {name}: {len(docs)} chunks, {len(questions)} questions ==")
        for q in (q for q in questions if q["answerable"]):
            ranks = {
                "vector": first_hit_rank(vector_rank(q["question"]), docs, q["start"], q["end"]),
                "hybrid": first_hit_rank(hybrid.rank(q["question"], args.k), docs, q["start"], q["end"]),
            }
            for mode, rank in ranks.items():
                t = totals[mode]
                t["n"] += 1
                t["rr"] += 1 / rank if rank else 0
                for c in cutoffs:
                    t[c] += bool(rank and rank <= c)
            gold = f"{format_timestamp(q['start'])}-{format_timestamp(q['end'])}"
            print(
                f"  vector rank {ranks['vector'] or '-'} | hybrid rank {ranks['hybrid'] or '-'}"
                f" | gold {gold} | {q['question']}"
            )

        if args.answers:
            # The real chat chain: hybrid top-4 plus neighbouring chunks.
            chain = _make_chain(HybridRetriever(store, docs, k=4).search)
            print("  -- answer / abstain check (Gemini) --")
            for q in questions:
                answer = chain.invoke(q["question"])["answer"].strip()
                abstained = answer == NOT_FOUND
                ok = abstained != q["answerable"]  # abstain exactly when unanswerable
                behaviour_ok += ok
                behaviour_total += 1
                print(f"  {'OK  ' if ok else 'FAIL'} {'abstained' if abstained else 'answered '}  {q['question']}")

    print("\n== Totals over both videos ==")
    for mode, t in totals.items():
        n = t["n"]
        parts = [f"recall@{c} = {t[c]}/{n} = {t[c] / n:.2f}" for c in cutoffs]
        print(f"{mode:6s} " + " | ".join(parts) + f" | MRR@{args.k} = {t['rr'] / n:.2f}")
    if args.answers:
        print(f"correct answer/abstain behaviour = {behaviour_ok}/{behaviour_total}")


if __name__ == "__main__":
    main()
