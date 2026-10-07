# AI Video Assistant

Paste a YouTube link (or a local audio/video file) and get a title, summary, action items, key decisions, open questions, and a chat that answers questions about the video, with every answer cited to the moment in the video it came from.

## What it does

- **Transcribes** the video. YouTube captions are used when available (about 2 seconds); otherwise the audio is downloaded and transcribed locally with Whisper (faster-whisper), or with Sarvam AI for Hinglish.
- **Summarises and extracts** with Google Gemini. The five analysis calls run in parallel.
- **Grounded chat (RAG).** The transcript is split into timestamped chunks, embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB. Each answer:
  - cites timestamps like `[06:44]`, which link to that second of the YouTube video;
  - shows the transcript excerpts it was given, so you can check it;
  - says "I could not find this information in the video transcript." when the video does not cover the question, instead of guessing.

## Retrieval quality

Measured on a 9.6-minute video with 13 hand-written questions (`eval/questions.json`). Ten have a hand-labelled time range containing the answer; three are about things the video never mentions.

| Metric | Result |
| --- | --- |
| recall@1 | 0.80 (8/10) |
| recall@3 | 0.90 (9/10) |
| recall@5 | 0.90 (9/10) |
| MRR@5 | 0.83 |
| Off-topic questions refused | 3/3 |
| Correct answer-or-abstain behaviour | 12/13 |

A question counts as a hit at k if any of the top-k retrieved chunks overlaps its labelled time range. The one miss ("Which company sponsored the video?") was a retrieval miss; the model then abstained rather than invent an answer. This is one video and a small question set, so treat the numbers as a baseline, not a benchmark.

Reproduce it:

```
python eval/eval_retrieval.py            # retrieval metrics, no API calls
python eval/eval_retrieval.py --answers  # also checks abstention (uses Gemini)
```

## Run it

Requirements: Python 3.13, ffmpeg on PATH, and Node.js or Deno (yt-dlp needs a JavaScript runtime for YouTube downloads).

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r Requirements.txt
.venv\Scripts\streamlit.exe run app.py
```

Then open http://localhost:8501. For a terminal version, run `python main.py`.

Create a `.env` file:

| Key | Needed | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Yes | Free key from Google AI Studio, used for all Gemini calls |
| `GEMINI_MODEL` | No | Default `gemini-flash-latest` |
| `GEMINI_FALLBACK_MODEL` | No | Default `gemini-flash-lite-latest`, used when the main model is busy |
| `WHISPER_MODEL` | No | Default `small`; `base` is faster and less accurate |
| `WHISPER_DEVICE` | No | Default `cpu`; `cuda` for an NVIDIA GPU |
| `SARVAM_API_KEY` | Hinglish only | Sarvam AI speech-to-text |

## Project layout

| Path | Role |
| --- | --- |
| `app.py` | Streamlit UI |
| `main.py` | Terminal version |
| `utils/captions.py` | Fetches YouTube captions (fast path) |
| `utils/audio_processor.py` | Downloads or converts audio and splits it into 5-minute chunks |
| `core/transcriber.py` | Whisper / Sarvam transcription with timestamps |
| `core/llm.py` | Gemini model with automatic fallback |
| `core/analysis.py` | Runs title, summary and extraction in parallel |
| `core/summarizer.py`, `core/extractor.py` | The individual Gemini prompts |
| `core/vector_store.py` | Timestamped chunks into ChromaDB |
| `core/rag_engine.py` | Retrieval, cited answers, abstention |
| `eval/` | Retrieval evaluation script, questions and the cached transcript |

## Known limits

- One video at a time: analysing a new video replaces the previous chat index.
- Retrieval is vector-only (no keyword search or reranker yet).
- Each chat question is answered on its own; follow-up questions do not see earlier turns.
- The extractors send the whole transcript in one call, so very long videos may exceed the model's context.
