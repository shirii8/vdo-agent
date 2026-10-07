# AI Video Assistant

Paste a YouTube link (or a local audio/video file) and get a title, summary, action items, key decisions, open questions, and a chat that answers questions about the video, with every answer cited to the moment in the video it came from.

## What it does

- **Transcribes** the video. YouTube captions are used when available (about 2 seconds); otherwise the audio is downloaded and transcribed locally with Whisper (faster-whisper), or with Sarvam AI for Hinglish.
- **Summarises and extracts** with Google Gemini. The five analysis calls run in parallel.
- **Interactive transcript.** The video and its transcript sit side by side: click a sentence to jump there, the sentence being spoken is highlighted as it plays, and a search box finds words in the transcript. It runs entirely in the browser.
- **Grounded chat (RAG).** The transcript is split into timestamped chunks, embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB. Retrieval is hybrid: BM25 keyword search and vector search are fused with Reciprocal Rank Fusion, and each hit brings its neighbouring chunks so the model reads whole passages. Each answer:
  - cites timestamps like `[06:44]`, which link to that second of the YouTube video;
  - shows the transcript excerpts it was given, so you can check it;
  - says "I could not find this information in the video transcript." when the video does not cover the question, instead of guessing.

## Retrieval quality

Measured on two videos (9.6 and 21 minutes) with 26 hand-written questions in `eval/`. Twenty have a hand-labelled time range containing the answer; six are about things the videos never mention.

| Metric (20 answerable questions) | Vector only | Hybrid (BM25 + vector) |
| --- | --- | --- |
| recall@1 | 0.80 (16/20) | 0.85 (17/20) |
| recall@3 | 0.95 (19/20) | 0.95 (19/20) |
| recall@5 | 0.95 (19/20) | 0.95 (19/20) |
| MRR@5 | 0.87 | 0.89 |

A question counts as a hit at k if any of the top-k retrieved chunks overlaps its labelled time range. Hybrid search is a small gain here; the bigger fix for answer quality was context. Chunks are about 30 seconds of speech, so a broad question such as "explain identity protocols" used to retrieve one fragment and get refused even though the video covers it. Adding each hit's neighbouring chunks, and refusing only when no excerpt is about the question, fixed that.

On a 10-question spot check through the full chat chain (4 answerable, 6 off-topic), 9 were handled correctly: all 6 off-topic questions were refused, and 3 of 4 answerable ones were answered. The miss ("Which company sponsored the video?") is a retrieval miss in both modes; the model abstains rather than invent an answer.

This is two videos and a small question set, so treat the numbers as a baseline, not a benchmark.

Reproduce it:

```
python eval/eval_retrieval.py            # vector vs hybrid retrieval, no API calls
python eval/eval_retrieval.py --answers  # also checks answer/abstain behaviour (26 Gemini calls)
```

## Run it

Requirements: Python 3.13, ffmpeg on PATH, and Node.js or Deno (yt-dlp needs a JavaScript runtime for YouTube downloads).

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\streamlit.exe run app.py
```

Then open http://localhost:8501. For a terminal version, run `python main.py`.

Create a `.env` file:

| Key | Needed | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Yes | Free key from Google AI Studio, used for all Gemini calls |
| `GEMINI_MODEL` | No | Default `gemini-flash-latest` |
| `GEMINI_FALLBACK_MODELS` | No | Comma-separated models tried in order when the one before is busy or out of quota |
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
| `core/retriever.py` | Hybrid retrieval: BM25 + vector search fused, plus neighbouring chunks |
| `core/rag_engine.py` | Chat chain: cited answers, abstention |
| `utils/transcript_view.py` | Interactive transcript widget (player, click-to-seek, search) |
| `eval/` | Retrieval evaluation script, questions and cached transcripts for two videos |

## Known limits

- One video at a time: analysing a new video replaces the previous chat index.
- No reranker yet: the fused hybrid ranking is used as is.
- Gemini's free tier limits each model per day (about 20 requests on the main Flash model). The app falls back through several models, but heavy use in one day can exhaust them.
- Each chat question is answered on its own; follow-up questions do not see earlier turns.
- The extractors send the whole transcript in one call, so very long videos may exceed the model's context.
