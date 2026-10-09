"""
Fast path for YouTube videos: fetch the captions YouTube already has instead
of downloading the audio and running Whisper. Takes ~2 seconds instead of
minutes, and the captions come with timestamps.

Two ways of getting them are tried in order, because YouTube often blocks one
of them on cloud servers:
  1. youtube-transcript-api  - a direct request for the caption track
  2. yt-dlp                  - reads the caption track listed in the video's
                               own page data

Returns None when neither works, so the caller can fall back to download +
Whisper.
"""

import json
from urllib.parse import urlparse, parse_qs

# English variants to accept, in order of preference.
CAPTION_LANGUAGES = ["en", "en-US", "en-GB", "en-IN"]


def youtube_id(source: str):
    """Video id from youtube.com/watch?v=ID, youtu.be/ID or /shorts/ID; else None."""
    url = urlparse(source.strip())
    host = url.netloc.lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        return url.path.strip("/") or None
    if host.endswith("youtube.com"):
        if url.path.startswith("/shorts/"):
            return url.path.split("/")[2]
        return parse_qs(url.query).get("v", [None])[0]
    return None


def _via_transcript_api(video_id: str):
    from youtube_transcript_api import YouTubeTranscriptApi

    transcript = YouTubeTranscriptApi().fetch(video_id, languages=CAPTION_LANGUAGES)
    return [
        {
            "start": snippet.start,
            "end": snippet.start + snippet.duration,
            # Captions contain line breaks and cues like "[Music]"; flatten the breaks.
            "text": snippet.text.replace("\n", " ").strip(),
        }
        for snippet in transcript
    ]


def _via_ytdlp(video_id: str):
    import yt_dlp

    options = {
        "quiet": True,
        "noplaylist": True,
        "skip_download": True,
        "js_runtimes": {"node": {}, "deno": {}},
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

        # Captions a person uploaded are preferred over auto-generated ones.
        track = None
        for tracks in (info.get("subtitles") or {}, info.get("automatic_captions") or {}):
            for language in CAPTION_LANGUAGES + ["en-orig"]:
                # "json3" is YouTube's JSON caption format with millisecond timings.
                track = next((f for f in tracks.get(language, []) if f.get("ext") == "json3"), None)
                if track:
                    break
            if track:
                break
        if not track:
            return []

        events = json.loads(ydl.urlopen(track["url"]).read().decode("utf-8")).get("events", [])

    # Keep events that carry words (auto captions also contain bare line breaks).
    lines = []
    for event in events:
        text = "".join(seg.get("utf8", "") for seg in event.get("segs") or [])
        text = text.replace("\n", " ").strip()
        if text:
            start = event.get("tStartMs", 0) / 1000
            lines.append({"start": start, "end": start + event.get("dDurationMs", 0) / 1000, "text": text})

    # Auto captions stay on screen while the next line appears, so their times
    # overlap; end each line where the next one starts.
    for current, following in zip(lines, lines[1:]):
        current["end"] = min(current["end"], following["start"])
    return lines


def fetch_youtube_captions(source: str):
    """Return [{"start", "end", "text"}, ...] from YouTube captions, or None."""
    video_id = youtube_id(source)
    if not video_id:
        return None

    for name, fetch in (("transcript API", _via_transcript_api), ("yt-dlp", _via_ytdlp)):
        try:
            segments = [s for s in fetch(video_id) if s["text"]]
        except Exception as error:
            # No captions, captions disabled, private video, blocked request, ...
            print(f"Captions via {name} failed ({type(error).__name__}).")
            continue
        if segments:
            print(f"Fetched {len(segments)} caption lines from YouTube via {name}.")
            return segments

    print("No YouTube captions available; falling back to Whisper.")
    return None
