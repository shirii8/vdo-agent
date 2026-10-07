"""
Fast path for YouTube videos: fetch the captions YouTube already has instead
of downloading the audio and running Whisper. Takes ~2 seconds instead of
minutes, and the captions come with timestamps.

Returns None when there are no usable captions (or the request is blocked),
so the caller can fall back to download + Whisper.
"""

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


def fetch_youtube_captions(source: str):
    """Return [{"start", "end", "text"}, ...] from YouTube captions, or None."""
    video_id = youtube_id(source)
    if not video_id:
        return None

    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript = YouTubeTranscriptApi().fetch(video_id, languages=CAPTION_LANGUAGES)
    except Exception as error:
        # No captions, captions disabled, private video, rate limit, ...
        print(f"No YouTube captions available ({type(error).__name__}); falling back to Whisper.")
        return None

    segments = [
        {
            "start": snippet.start,
            "end": snippet.start + snippet.duration,
            # Captions contain line breaks and cues like "[Music]"; flatten the breaks.
            "text": snippet.text.replace("\n", " ").strip(),
        }
        for snippet in transcript
    ]
    segments = [s for s in segments if s["text"]]
    print(f"Fetched {len(segments)} caption lines from YouTube.")
    return segments or None
