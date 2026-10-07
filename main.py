"""
Command-line version of the AI Video Assistant.

Runs the full pipeline (audio -> transcript -> title/summary/extractions -> RAG)
and then opens an interactive chat loop in the terminal.
Usage:  python main.py
"""

from dotenv import load_dotenv

load_dotenv()  # load .env into os.environ BEFORE core/ modules read it

from utils.audio_processor import process_input
from utils.captions import fetch_youtube_captions
from core.transcriber import transcribe_segments, segments_to_text
from core.analysis import analyze_transcript
from core.rag_engine import build_rag_chain, ask_question, format_timestamp


def run_pipeline(source: str, language: str = "english") -> dict:
    """Run every stage on one video and return all results in a dict."""
    print("starting AI Video Assistant")

    # Fast path: English YouTube videos usually have captions (seconds, not minutes).
    segments = fetch_youtube_captions(source) if language == "english" else None

    if not segments:
        # 1. URL/file -> list of 5-minute WAV chunks
        chunks = process_input(source)

        # 2. Chunks -> timestamped segments (Whisper or Sarvam)
        segments = transcribe_segments(chunks, language)

    transcript = segments_to_text(segments)
    print(f"raw transcription (first 300 characters ) {transcript[:300]}")

    # 3. LLM analysis with Gemini: title, summary and extractions in parallel
    analysis = analyze_transcript(transcript)

    # 4. Index the timestamped segments in Chroma for question answering
    rag_chain = build_rag_chain(segments)

    return {
        **analysis,
        "transcript": transcript,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\nChat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue
        result = ask_question(rag_chain, question)
        print(f"\nAssistant: {result['answer']}")
        # Show which transcript excerpts the answer was grounded in.
        for doc in result["sources"]:
            stamp = format_timestamp(doc.metadata.get("start", 0))
            print(f"   [{stamp}] {doc.page_content[:120]}...")
        print()
