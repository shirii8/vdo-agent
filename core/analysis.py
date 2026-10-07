"""
Runs all the LLM analysis on a transcript at the same time.

Title, summary, action items, key decisions and open questions don't depend
on each other, so they are sent to Gemini in parallel threads. Total time is
roughly that of the slowest call instead of the sum of all five.
"""

from concurrent.futures import ThreadPoolExecutor

from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)

# Result key -> function that produces it from the transcript.
TASKS = {
    "title": generate_title,
    "summary": summarize,
    "action_items": extract_action_items,
    "key_decisions": extract_key_decisions,
    "open_questions": extract_questions,
}


def analyze_transcript(transcript: str) -> dict:
    """Return {"title", "summary", "action_items", "key_decisions", "open_questions"}."""
    with ThreadPoolExecutor(max_workers=len(TASKS)) as pool:
        futures = {key: pool.submit(fn, transcript) for key, fn in TASKS.items()}
        # .result() re-raises any exception from the thread, so errors still surface.
        return {key: future.result() for key, future in futures.items()}
