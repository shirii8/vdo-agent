"""
Measures how good the RAG retrieval is on a known video.

Each question in questions.json has a hand-labelled "gold" time range: the part
of the video that contains the answer. A question counts as a hit at k if any
of the top-k retrieved chunks overlaps that range.

  recall@k = hits / answerable questions
  MRR      = mean of 1/rank of the first correct chunk (0 if not in top k)

With --answers it also calls Gemini and checks abstention: questions marked
"answerable": false must get the "could not find" reply, and answerable ones
must not.

Usage (from the project root):
    python eval/eval_retrieval.py              # retrieval only, no API calls
    python eval/eval_retrieval.py --answers    # + abstention check (uses Gemini)
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
from core.rag_engine import _make_chain, format_timestamp, NOT_FOUND

HERE = os.path.dirname(os.path.abspath(__file__))


def overlaps(doc, gold_start: float, gold_end: float) -> bool:
    """True if the chunk's time span intersects the gold time range."""
    return doc.metadata["start"] <= gold_end and doc.metadata["end"] >= gold_start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--answers", action="store_true", help="also check abstention with Gemini")
    args = parser.parse_args()

    segments = json.load(open(os.path.join(HERE, "ai_economy_segments.json"), encoding="utf-8"))["segments"]
    questions = json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))

    # In-memory collection, so the app's own vector_db/ is left untouched.
    docs = segments_to_documents(segments)
    store = Chroma.from_documents(docs, get_embeddings(), collection_name="eval_retrieval")
    retriever = store.as_retriever(search_kwargs={"k": args.k})
    chain = _make_chain(retriever) if args.answers else None

    answerable = [q for q in questions if q["answerable"]]
    hits = {1: 0, 3: 0, args.k: 0}
    reciprocal_ranks = []

    print(f"{len(docs)} chunks indexed, {len(questions)} questions, k={args.k}\n")
    for q in answerable:
        retrieved = retriever.invoke(q["question"])
        # 1-based rank of the first chunk that overlaps the gold range, or None.
        rank = next(
            (i + 1 for i, d in enumerate(retrieved) if overlaps(d, q["start"], q["end"])), None
        )
        for cutoff in hits:
            hits[cutoff] += bool(rank and rank <= cutoff)
        reciprocal_ranks.append(1 / rank if rank else 0)
        gold = f"{format_timestamp(q['start'])}-{format_timestamp(q['end'])}"
        print(f"  {'HIT ' if rank else 'MISS'} rank={rank or '-'}  gold {gold}  {q['question']}")

    n = len(answerable)
    print()
    for cutoff, count in hits.items():
        print(f"recall@{cutoff} = {count}/{n} = {count / n:.2f}")
    print(f"MRR@{args.k}   = {sum(reciprocal_ranks) / n:.2f}")

    if chain:
        print("\nAbstention check (Gemini):")
        correct = 0
        for q in questions:
            answer = chain.invoke(q["question"])["answer"].strip()
            abstained = answer == NOT_FOUND
            ok = abstained != q["answerable"]  # should abstain exactly when unanswerable
            correct += ok
            print(f"  {'OK  ' if ok else 'FAIL'} {'abstained' if abstained else 'answered '}  {q['question']}")
        print(f"\ncorrect answer/abstain behaviour = {correct}/{len(questions)}")


if __name__ == "__main__":
    main()
