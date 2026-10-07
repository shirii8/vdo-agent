"""
Turns the user's input (YouTube URL or local audio/video file) into a list of
WAV files of at most 10 minutes each, ready for transcription.

Requires ffmpeg on PATH (used by both yt-dlp and pydub).
"""

import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloades"  # where YouTube audio is saved
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """Download the best audio stream of a YouTube video and convert it to WAV."""
    # Name the file by video id: titles can contain characters that are
    # awkward in file names.
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        # A link copied from a playlist (…&list=…) must download only that one
        # video, not the whole playlist.
        "noplaylist": True,
        # After download, ffmpeg re-encodes the stream to .wav.
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        # Write mono 16 kHz audio: all the speech models need, and about a
        # sixth of the size of the default stereo 48 kHz WAV (less RAM later).
        "postprocessor_args": {"extractaudio": ["-ac", "1", "-ar", "16000"]},
        "quiet": True,
        # YouTube now requires solving a JavaScript challenge; without a JS
        # runtime the stream URLs are rejected with HTTP 403. Node.js and Deno
        # are both allowed (needs Node or Deno installed and on PATH).
        "js_runtimes": {"node": {}, "deno": {}},
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            fallback = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
    except yt_dlp.utils.DownloadError as error:
        # Typical on cloud hosts: YouTube answers "Sign in to confirm you're
        # not a bot" to data-centre IP addresses.
        raise RuntimeError(f"YouTube refused the audio download: {error}") from error

    # yt-dlp records the final (post-conversion) path of what it downloaded.
    downloads = info.get("requested_downloads") or []
    filename = downloads[0].get("filepath") if downloads else fallback

    if not filename or not os.path.exists(filename):
        raise FileNotFoundError(
            "The audio download did not produce a file "
            f"(got a {info.get('_type', 'video')} with {len(downloads)} download(s); "
            f"expected {filename or fallback}; "
            f"folder holds {sorted(os.listdir(DOWNLOAD_DIR))[:5]}). "
            "YouTube may be blocking this server."
        )
    return filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    # Mono, 16 kHz: the format speech models expect; also shrinks the file.
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 5) -> list:
    """Split a WAV into consecutive chunks and return the chunk file paths."""
    # 5-minute chunks keep Whisper's peak memory low (it pre-processes a whole
    # chunk at once), which matters on machines with little free RAM.
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000  # pydub works in milliseconds

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    """Entry point: URL -> download, file path -> convert; then chunk."""
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
