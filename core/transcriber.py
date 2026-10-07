"""
Speech-to-text for the audio chunks produced by utils/audio_processor.py.

Two engines, picked by the `language` argument:
  - "english"  -> Whisper via faster-whisper, running locally (WHISPER_MODEL)
  - "hinglish" -> Sarvam AI cloud API, which transcribes AND translates to English

Both return *segments* — {"start", "end", "text"} with times in seconds from
the start of the whole video — so the chat can cite where an answer came from.
"""

import os
import wave
import numpy as np
import requests
from pydub import AudioSegment

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

# tiny / base / small / medium / large-v3 — bigger is more accurate but slower.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# "cpu" (default) or "cuda" for an NVIDIA GPU (needs CUDA 12 + cuDNN 9 libraries).
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

# Cached Whisper model so it is loaded from disk only once per process.
_model = None


def load_model():
    global _model

    if _model is None:
        from faster_whisper import WhisperModel

        # int8 on CPU: ~4x faster than plain Whisper and uses far less RAM.
        compute_type = "float16" if WHISPER_DEVICE == "cuda" else "int8"
        print(f"Loading Whisper model: {WHISPER_MODEL} ({WHISPER_DEVICE}/{compute_type}) ...")
        _model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=compute_type)
        print("Whisper model loaded.")
    return _model


def _load_audio(chunk_path: str) -> np.ndarray:
    """Read a WAV as mono 16 kHz float32 samples in [-1, 1] (what Whisper expects)."""
    audio = AudioSegment.from_wav(chunk_path).set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.set_sample_width(2).get_array_of_samples(), dtype=np.float32)
    return samples / 32768.0


def transcribe_chunk_whisper(chunk_path: str) -> list:
    model = load_model()

    # beam_size=1 (greedy) is much faster with little accuracy loss;
    # vad_filter skips silence so it isn't transcribed (or hallucinated).
    segments, _info = model.transcribe(_load_audio(chunk_path), beam_size=1, vad_filter=True)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        # multipart/form-data upload: the audio file plus model options.
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n X Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> list:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces and send each separately; each piece becomes one segment.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000  # pydub slices in milliseconds

    segments = []
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms  # ceiling division

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start : start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            text = _send_to_sarvam(piece_path).strip()
            if text:
                segments.append(
                    {"start": start / 1000, "end": (start + len(piece)) / 1000, "text": text}
                )
        finally:
            # Always delete the temporary piece, even if the request failed.
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return segments


def transcribe_chunk(chunk_path: str, language: str = "english") -> list:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    Segment times are relative to the start of this chunk.
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_segments(chunks: list, language: str = "english") -> list:
    """Transcribe every chunk in order; return segments timed from the video start."""
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    all_segments = []
    offset = 0.0  # seconds of audio in the chunks before this one

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        for seg in transcribe_chunk(chunk, language=language):
            all_segments.append(
                {"start": seg["start"] + offset, "end": seg["end"] + offset, "text": seg["text"]}
            )

        # Read the chunk's length from its WAV header (no need to load the audio).
        with wave.open(chunk, "rb") as w:
            offset += w.getnframes() / w.getframerate()

    print("Transcription complete.")
    return all_segments


def segments_to_text(segments: list) -> str:
    """Join segments into one plain transcript string (for summary/extraction)."""
    return " ".join(seg["text"] for seg in segments).strip()


def transcribe_all(chunks: list, language: str = "english") -> str:
    """Transcribe every chunk and return one plain transcript string."""
    return segments_to_text(transcribe_segments(chunks, language))
