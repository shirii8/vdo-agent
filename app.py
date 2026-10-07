"""
Streamlit web UI for the AI Video Assistant.  Run:  streamlit run app.py

Streamlit re-executes this whole script top to bottom on every interaction
(button click, text input). Anything that must survive between runs — the
pipeline result, chat history — is kept in st.session_state.
"""

from dotenv import load_dotenv

load_dotenv()  # load .env into os.environ BEFORE core/ modules read it

import html
from markdown_it import MarkdownIt
import os
import streamlit as st
import time
from utils.audio_processor import process_input
from core.transcriber import transcribe_segments, segments_to_text
from core.analysis import analyze_transcript
from core.rag_engine import build_rag_chain, ask_question, format_timestamp, NOT_FOUND
from utils.captions import fetch_youtube_captions, youtube_id
from utils.transcript_view import build_transcript_html
from utils.mascot import rewi_svg
import streamlit.components.v1 as components
import re

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rewi · AI Video Assistant",
    # Rewi as the browser-tab icon (path is relative to this file, not the cwd).
    page_icon=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "rewi-favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=DM+Sans:wght@400;500;700&display=swap');

/* ── Palette: Buttermilk #FFF1B5 · Pastel Blue #C1DBE8 · Old Burgundy #43302E ── */
:root {
    --bg: #FFF9E3;            /* page: light buttermilk */
    --surface: #FFFDF6;       /* cards */
    --surface-2: #FFF1B5;     /* buttermilk: transcript box, user bubbles */
    --border: #E8D9A0;        /* darker buttermilk for borders */
    --accent: #43302E;        /* old burgundy: buttons, titles, emphasis */
    --accent-glow: #5C4441;   /* lighter burgundy for hover */
    --accent-2: #C1DBE8;      /* pastel blue: sidebar, assistant bubbles */
    --accent-2-deep: #8DB8CE; /* deeper pastel blue for lines and dots */
    --text: #43302E;
    --text-muted: #7A6461;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
}

/* Tighter top: Streamlit leaves ~6rem above the content and an empty header
   strip above the sidebar logo. */
[data-testid="stMainBlockContainer"], .block-container {
    padding-top: 2.25rem !important;
}
/* The sidebar header only holds the collapse arrow: float it in the top-right
   corner so it no longer pushes the logo down. */
[data-testid="stSidebarHeader"] {
    position: absolute !important;
    top: 0.6rem !important;
    right: 0.6rem !important;
    height: auto !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    z-index: 2;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 1.5rem !important;
}
/* The top bar stays (it holds the sidebar re-open arrow) but is see-through. */
[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Sidebar: pastel blue panel ── */
[data-testid="stSidebar"] {
    background: var(--accent-2) !important;
    border-right: 1px solid var(--accent-2-deep) !important; padding: 0.5rem
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Cormorant Garamond', serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.2rem, 5vw, 3.6rem);
    font-weight: 700;
    line-height: 1.05;
    margin: 0;
    color: var(--accent);
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.5rem 1.5rem 1.5rem 1.75rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}

.card:hover {
    border-color: var(--accent-2-deep);
    box-shadow: 0 6px 20px rgba(67, 48, 46, 0.08);
}

/* pastel-blue stripe on the left edge */
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 6px; height: 100%;
    background: var(--accent-2);
}

.card-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.92rem;
    line-height: 1.7;
    color: var(--text);
}

/* Markdown rendered by md(): tighten default paragraph/list spacing */
.card-content p, .chat-bubble p { margin: 0 0 0.5rem 0; }
.card-content ul, .card-content ol,
.chat-bubble ul, .chat-bubble ol { margin: 0 0 0.5rem 0; padding-left: 1.25rem; }
.card-content li, .chat-bubble li { margin-bottom: 0.25rem; }
.card-content > :last-child, .chat-bubble > :last-child { margin-bottom: 0; }
.card-content strong, .chat-bubble strong { font-weight: 700; color: var(--accent); }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-purple { background: var(--accent);    color: var(--surface-2) !important; border: 1px solid var(--accent); }
.badge-cyan   { background: var(--accent-2);  color: var(--accent);    border: 1px solid var(--accent-2-deep); }
.badge-green  { background: var(--surface-2); color: var(--accent);    border: 1px solid var(--border); }

/* ── Inputs & Buttons ── */
/* Text input: ONE border, on Streamlit's outer wrapper. The inner layers are
   made transparent and borderless so no second outline shows through. */
.stTextInput [data-testid="stTextInputRootElement"],
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.15s ease !important;
}

.stTextInput input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stTextInput input::placeholder { color: var(--text-muted) !important; opacity: 0.7; }

.stTextInput [data-testid="stTextInputRootElement"]:focus-within {
    border-color: var(--accent) !important;
}

/* Hide the "Press Enter to apply" hint that overlaps the text. The value is
   also applied when the field loses focus, e.g. on clicking a button. */
[data-testid="InputInstructions"] { display: none !important; }

.stButton > button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 999px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.65rem 1.5rem !important;
    /* Flat button: only colour and a slight press animate, no shadow or lift. */
    box-shadow: none !important;
    transition: background-color 0.15s ease, transform 0.08s ease !important;
    text-transform: uppercase !important;
}

/* button label: buttermilk on burgundy. The extra selectors out-rank the
   sidebar "*" rule and the markdown "p" rule further down. */
.stButton > button,
.stButton > button *,
[data-testid="stSidebar"] .stButton > button *,
.stButton > button [data-testid="stMarkdownContainer"] p {
    color: var(--surface-2) !important;
}

/* Hover deepens the burgundy (a lighter brown looked washed out). */
.stButton > button:hover {
    background: #2F201E !important;
    box-shadow: none !important;
}

/* Pressed: a small squeeze so the click feels acknowledged. */
.stButton > button:active {
    background: #2F201E !important;
    transform: scale(0.98) !important;
    box-shadow: none !important;
}

/* Keyboard focus: a clear ring instead of Streamlit's default red glow. */
.stButton > button:focus,
.stButton > button:focus-visible {
    box-shadow: none !important;
    outline: none !important;
}
.stButton > button:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px !important;
}

.stButton > button:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
}

/* Secondary button (Clear Chat): pastel blue with burgundy text */
.stButton > button[kind="secondary"] {
    background: var(--accent-2) !important;
    border: 1px solid var(--accent-2-deep) !important;
}

.stButton > button[kind="secondary"]:hover,
.stButton > button[kind="secondary"]:active {
    background: var(--accent-2-deep) !important;
}

.stButton > button[kind="secondary"],
.stButton > button[kind="secondary"] *,
[data-testid="stSidebar"] .stButton > button[kind="secondary"] *,
.stButton > button[kind="secondary"] [data-testid="stMarkdownContainer"] p {
    color: var(--accent) !important;
}

/* ── Progress / Status ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.7rem 1rem;
    background: var(--surface);
    border-radius: 12px;
    margin: 0.4rem 0;
    border: 1px solid var(--accent-2-deep);
    font-size: 0.8rem;
}

.status-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-active   { background: var(--accent-2-deep); box-shadow: 0 0 0 3px var(--accent-2); animation: pulse 1.5s infinite; }
.dot-done     { background: var(--accent); }
.dot-pending  { background: var(--border); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chat ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.chat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.chat-bubble {
    display: inline-block;
    padding: 0.65rem 1rem;
    border-radius: 16px;
    font-size: 0.88rem;
    line-height: 1.6;
    max-width: 90%;
    color: var(--text);
}

.user-label  { color: var(--accent); }
.bot-label   { color: var(--text-muted); }

/* user = buttermilk, assistant = pastel blue */
.user-bubble { background: var(--surface-2); border: 1px solid var(--border);        align-self: flex-end; }
.bot-bubble  { background: var(--accent-2);  border: 1px solid var(--accent-2-deep); align-self: flex-start; }

/* Timestamp citations: burgundy pills that open the video at that moment */
.chat-bubble a {
    color: var(--surface-2) !important;
    background: var(--accent);
    border-radius: 999px;
    padding: 0 0.45rem;
    font-size: 0.75rem;
    font-weight: 700;
    text-decoration: none;
    white-space: nowrap;
}
.chat-bubble a:hover { background: #2F201E; }
/* small play triangle before each timestamp */
.chat-bubble a::before {
    content: '';
    display: inline-block;
    margin-right: 0.3rem;
    border-left: 6px solid currentColor;
    border-top: 4px solid transparent;
    border-bottom: 4px solid transparent;
}

/* Retrieved excerpts under each answer */
.sources { margin-top: 0.6rem; border-top: 1px solid var(--accent-2-deep); padding-top: 0.5rem; }
.sources summary { cursor: pointer; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-muted); }
.source { font-size: 0.8rem; line-height: 1.5; margin-top: 0.5rem; padding: 0.5rem 0.65rem; background: var(--surface); border-radius: 10px; color: var(--text); }
.source-time { margin-right: 0.35rem; font-weight: 700; }

/* ── Rewi, the mascot (shapes come from utils/mascot.py) ── */
.rewi .rw-b  { fill: var(--accent); }
.rewi .rw-bs { stroke: var(--accent); fill: none; }
.rewi .rw-b.rw-bs { fill: var(--accent); }
.rewi .rw-e  { fill: var(--bg); }
.rewi .rw-es { stroke: var(--bg); fill: none; }
.rewi .rw-a  { fill: var(--accent-2); }
.rewi .rw-ring { fill: var(--accent-2); stroke: var(--accent); }
.rewi { overflow: visible; }

.rewi .rw-bob     { animation: rewi-bob 1.6s ease-in-out infinite alternate; }
.rewi .rw-antenna { transform-box: view-box; transform-origin: 80px 42px; animation: rewi-wiggle 2.4s ease-in-out infinite; }
.rewi .rw-eye     { transform-box: fill-box; transform-origin: center; animation: rewi-blink 4s infinite; }
.rewi .rw-cheek   { animation: rewi-blush 3.2s ease-in-out infinite alternate; }
/* listening: sound waves pulse outwards, inner pair first */
.rewi .rw-wave    { animation: rewi-wave 1.2s ease-in-out infinite; }
.rewi .rw-wave-2  { animation-delay: 0.3s; }
/* thinking: three dots bounce one after another */
.rewi .rw-dot     { transform-box: fill-box; transform-origin: center; animation: rewi-dot 1.2s ease-in-out infinite; }
.rewi .rw-dot-2   { animation-delay: 0.15s; }
.rewi .rw-dot-3   { animation-delay: 0.3s; }
/* done: the check badge pops in once */
.rewi .rw-badge   { transform-box: fill-box; transform-origin: center; animation: rewi-pop 0.45s ease-out both; }
/* a still Rewi (the logo) */
.rewi.rw-still * { animation: none !important; }

@keyframes rewi-bob    { from { transform: translateY(0); } to { transform: translateY(-6px); } }
@keyframes rewi-wiggle { 0%, 100% { transform: rotate(0deg); } 25% { transform: rotate(-9deg); } 60% { transform: rotate(7deg); } }
@keyframes rewi-blink  { 0%, 90%, 100% { transform: scaleY(1); } 94% { transform: scaleY(0.1); } }
@keyframes rewi-blush  { from { opacity: 0.7; } to { opacity: 1; } }
@keyframes rewi-wave   { 0%, 100% { opacity: 0.15; } 50% { opacity: 1; } }
@keyframes rewi-dot    { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-7px); } }
@keyframes rewi-pop    { from { transform: scale(0); } 70% { transform: scale(1.15); } to { transform: scale(1); } }
@media (prefers-reduced-motion: reduce) { .rewi * { animation: none !important; } }

/* Loader shown while a recording is analysed: Rewi, a headline, finished steps */
.loader { display: flex; flex-direction: column; align-items: center; text-align: center; padding: 2.5rem 1rem 1.5rem 1rem; }
.loader-title { font-family: 'Cormorant Garamond', serif; font-size: 1.6rem; font-weight: 700; color: var(--text); margin-top: 0.75rem; }
.loader-caption { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.15rem; }
.loader-steps { margin-top: 1.25rem; font-size: 0.78rem; color: var(--text-muted); line-height: 1.8; }
.loader-steps span { display: inline-block; padding: 0.1rem 0.7rem; margin: 0.15rem; border: 1px solid var(--border); border-radius: 999px; background: var(--surface); }

/* Rows inside the sidebar's "What you get" section */
.feature { display: flex; gap: 1rem; align-items: flex-start; font-size: 0.78rem; line-height: 1.45; margin-bottom: 0.6rem; margin-left: 0.5rem }
.feature:last-child { margin-bottom: 0.5rem; }
.feature svg { margin-top: 0.2rem; }

/* Collapsed sections in the sidebar (Options, Run details) */
[data-testid="stSidebar"] [data-testid="stExpander"] details {
    background: transparent !important;
    border: 1px solid var(--accent-2-deep) !important;
    border-radius: 12px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary { font-size: 0.8rem !important; padding: 0.5rem 0.75rem !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { color: var(--accent) !important; }

/* One quiet line shown instead of empty meeting-notes cards */
.quiet-note { font-size: 0.82rem; color: var(--text-muted); padding: 0.75rem 1rem; border: 1px dashed var(--border); border-radius: 12px; margin-bottom: 1rem; }

/* File uploader: cream drop zone with a dashed edge, matching the inputs */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1px dashed var(--accent-2-deep) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: var(--accent-2) !important;
    border: 1px solid var(--accent-2-deep) !important;
    border-radius: 999px !important;
    box-shadow: none !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    font-size: 0.85rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Streamlit built-ins ── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* Scrollbars are hidden everywhere; wheel, touch and keyboard scrolling still work. */
* { scrollbar-width: none !important; -ms-overflow-style: none !important; }
*::-webkit-scrollbar { display: none !important; width: 0 !important; height: 0 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Session State Init ──────────────────────────────────────────────────────────
# Create each key once; later reruns keep the existing values.
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
    "step_times": {},  # seconds each stage took, shown in the sidebar
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Helpers ────────────────────────────────────────────────────────────────────
# CommonMark renderer with raw HTML disabled, so model output can't inject tags.
_md = MarkdownIt("commonmark", {"html": False})


def md(text: str) -> str:
    """Render the model's Markdown (bold, numbered/nested lists) to safe HTML."""
    # Newlines removed so Streamlit's own Markdown pass can't end the HTML block early.
    rendered = _md.render(str(text)).replace("\n", "")
    # Open links (timestamp citations) in a new tab instead of inside the app.
    return rendered.replace("<a href=", '<a target="_blank" rel="noopener" href=')


def esc(text: str) -> str:
    """Escape model/transcript text before putting it inside raw HTML."""
    return html.escape(str(text))


# Line icons (24x24, drawn with the current text colour) used instead of emojis.
ICONS = {
    "play": '<circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/>',
    "audio": '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "tag": '<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/>',
    "list": '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "bookmark": '<path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "key": '<circle cx="7.5" cy="15.5" r="5.5"/><path d="M11.4 11.6L21 2"/><path d="M16 7l3 3"/><path d="M19 4l2 2"/>',
    "question": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
}


def icon(name: str, size: int = 15) -> str:
    """Inline SVG for one of ICONS; inherits colour from the surrounding text."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round" style="flex-shrink:0;vertical-align:-0.15em">{ICONS[name]}</svg>'
    )


UPLOAD_DIR = "uploads"  # where uploaded recordings are saved before processing


def save_upload(uploaded) -> str:
    """Write an uploaded recording to disk and return its path."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    # Keep only safe characters so the name works as a file name everywhere.
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", uploaded.name)
    path = os.path.join(UPLOAD_DIR, safe_name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def nothing_found(text: str) -> bool:
    """True for the extractor's 'No action items found.' style replies."""
    cleaned = str(text).strip().lower().strip("*_ .")
    return len(cleaned) < 80 and cleaned.startswith("no ") and "found" in cleaned


def time_link(source: str, seconds: float):
    """Link that opens the YouTube video at `seconds`; None for local files."""
    vid = youtube_id(source)
    return f"https://www.youtube.com/watch?v={vid}&t={int(seconds)}s" if vid else None


# Matches citations the model writes, e.g. [03:12] or [1:02:05].
_CITATION = re.compile(r"\[(\d{1,2}(?::\d{2}){1,2})\]")


def _to_seconds(stamp: str) -> int:
    seconds = 0
    for part in stamp.split(":"):
        seconds = seconds * 60 + int(part)
    return seconds


def link_citations(answer: str, source: str) -> str:
    """Turn [mm:ss] citations into Markdown links that jump to that moment."""
    if not youtube_id(source):
        return answer
    return _CITATION.sub(
        lambda m: f"[{m.group(1)}]({time_link(source, _to_seconds(m.group(1)))})", answer
    )


def sources_html(sources: list, source: str, answered: bool) -> str:
    """Collapsible list of the transcript excerpts the answer was grounded in."""
    if not sources:
        return ""
    label = (
        f"Sources · {len(sources)} transcript excerpts"
        if answered
        else f"Closest excerpts searched · {len(sources)} (none contained the answer)"
    )
    items = ""
    for src in sources:
        stamp = format_timestamp(src["start"])
        href = time_link(source, src["start"])
        stamp_html = (
            f'<a href="{esc(href)}" target="_blank" rel="noopener">{stamp}</a>'
            if href
            else stamp
        )
        snippet = src["text"] if len(src["text"]) <= 260 else src["text"][:260] + "…"
        items += f'<div class="source"><span class="source-time">{stamp_html}</span> {esc(snippet)}</div>'
    return f'<details class="sources"><summary>{label}</summary>{items}</details>'


def step_status(steps: dict, key: str) -> str:
    """Map a step state (pending/active/done) to the CSS class of its dot."""
    s = steps.get(key, "pending")
    if s == "active":
        return "dot-active"
    if s == "done":
        return "dot-done"
    return "dot-pending"


def render_step_bar(label: str, key: str, icon_name: str):
    """Draw one row of the pipeline status list in the sidebar."""
    css = step_status(st.session_state.pipeline_steps, key)
    secs = st.session_state.step_times.get(key)
    timing = f" · {secs:.0f}s" if secs is not None else ""
    st.markdown(
        f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span style="display:flex;align-items:center;gap:0.45rem">{icon(icon_name, 14)}<span>{label}{timing}</span></span>
    </div>""",
        unsafe_allow_html=True,
    )


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div class="hero-title" style="font-size:1.9rem;display:flex;align-items:center;gap:0.55rem">{rewi_svg("happy", 40, animated=False)}<span>Rewi</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True
    )
    st.markdown("---")

    # The sidebar holds only what is needed to start a run; everything else is
    # tucked into collapsed sections so the first screen stays quiet.
    source = st.text_input(
        "YouTube link",
        placeholder="Paste a YouTube link or file path",
    )

    # A meeting recording can be uploaded instead of pasting a link or path.
    uploaded = st.file_uploader(
        "Or upload a recording",
        type=["mp3", "mp4", "m4a", "wav", "webm", "mov", "mkv", "ogg", "flac", "aac"],
        help="Audio or video up to 200 MB. It is transcribed here, which takes roughly "
        "as long as the recording itself.",
    )

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    with st.expander("Options"):
        # Captions are fetched in seconds; Whisper takes about as long as the video.
        use_captions = st.checkbox(
            "Use YouTube captions",
            value=True,
            help="Fast: skips the audio download and Whisper when the video has "
            "English captions. Untick to always transcribe the audio.",
        )

    run_btn = st.button("Analyse", type="primary", use_container_width=True)

    # The LLM steps all need this key; fail early with a clear message.
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("GOOGLE_API_KEY is missing from .env — summary, extraction and chat will fail.")

    # What the analysis produces, for first-time users; collapsed by default.
    with st.expander("What you get"):
        st.markdown(
            "".join(
                f'<div class="feature">{icon(name, 15)}<span><strong>{title}</strong><br>{text}</span></div>'
                for name, title, text in [
                    ("list", "Summary", "Title and key points"),
                    ("check", "Meeting notes", "Tasks, decisions, open questions"),
                    ("file", "Interactive transcript", "Click a line to jump there"),
                    ("chat", "Cited chat", "Answers link to the moment"),
                ]
            ),
            unsafe_allow_html=True,
        )

    # After a run: one collapsed line with the total time; stage timings inside.
    if st.session_state.pipeline_done:
        total = st.session_state.get("run_seconds")
        label = f"Run details · {total:.0f}s" if total else "Run details"
        with st.expander(label):
            for step, icon_name, step_label in [
                ("audio", "audio", "Audio Processing"),
                ("transcript", "file", "Transcription"),
                ("title", "tag", "Title Generation"),
                ("summary", "list", "Summarisation"),
                ("extract", "search", "Extraction"),
                ("rag", "database", "RAG Engine"),
            ]:
                render_step_bar(step_label, step, icon_name)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
# Runs only on the rerun triggered by the Analyse click.
if run_btn:
    if uploaded is None and not source.strip():
        st.error("Paste a YouTube link or file path, or upload a recording.")
    else:
        # Clear the previous video's results before starting.
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        st.session_state.step_times = {}

        # An uploaded recording wins over the text field: save it to disk and
        # treat it like any other local file from here on.
        if uploaded is not None:
            source = save_upload(uploaded)

        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            # Live loader in the main area: Rewi changes expression with the stage,
            # and each finished stage is listed underneath with its time.
            run_started = time.time()
            finished = []  # "Transcribing 23s" pills for stages already done

            def show_loader(mood, title, caption):
                steps = "".join(f"<span>{esc(line)}</span>" for line in finished)
                progress_placeholder.markdown(
                    f'''<div class="loader">{rewi_svg(mood, 150)}
                    <div class="loader-title">{title}</div>
                    <div class="loader-caption">{caption}</div>
                    <div class="loader-steps">{steps}</div></div>''',
                    unsafe_allow_html=True,
                )

            def run_stage(label, keys, fn, *args, mood="thinking", title="Thinking", caption=""):
                """Show Rewi for this stage, run fn, and record how long it took."""
                for k in keys:
                    update_step(k, "active")
                show_loader(mood, title, caption or label)
                start = time.time()
                out = fn(*args)
                secs = time.time() - start
                for k in keys:
                    update_step(k, "done")
                    st.session_state.step_times[k] = secs
                finished.append(f"{label} {secs:.0f}s")
                return out

            # Fast path: English YouTube videos usually have captions with timestamps.
            segments = None
            if use_captions and language == "english" and youtube_id(source):
                segments = run_stage(
                    "Fetching YouTube captions",
                    ["audio", "transcript"],
                    fetch_youtube_captions,
                    source,
                    mood="listening",
                    title="Listening",
                    caption="Fetching the captions",
                )

            # Fallback (no captions, local file, Hinglish): download + transcribe.
            if not segments:
                chunks = run_stage(
                    "Downloading / converting audio",
                    ["audio"],
                    process_input,
                    source,
                    mood="listening",
                    title="Listening",
                    caption="Getting the audio ready",
                )
                # Timestamped segments: plain text for analysis, timings for citations.
                segments = run_stage(
                    "Transcribing",
                    ["transcript"],
                    transcribe_segments,
                    chunks,
                    language,
                    mood="listening",
                    title="Listening",
                    caption="Transcribing your recording",
                )
            transcript = segments_to_text(segments)
            if not transcript:
                raise ValueError("No speech was detected in this audio, so there is nothing to analyse.")

            # Title, summary and the three extractions run in parallel (core/analysis.py).
            analysis = run_stage(
                "Title, summary and extraction",
                ["title", "summary", "extract"],
                analyze_transcript,
                transcript,
                title="Thinking",
                caption="Writing the summary",
            )
            rag_chain = run_stage(
                "Building chat index",
                ["rag"],
                build_rag_chain,
                segments,
                title="Thinking",
                caption="Getting ready for your questions",
            )

            st.session_state.result = {
                **analysis,
                "transcript": transcript,
                "segments": segments,  # timed lines for the interactive transcript
                "source": source.strip(),  # used to build timestamp links
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            st.session_state.run_seconds = time.time() - run_started
            show_loader("done", "Done!", "Notes ready — ask me anything")
            time.sleep(1.2)  # let the wink and check badge be seen
            progress_placeholder.empty()
            st.rerun()  # redraw the page so the results section renders

        except Exception as e:
            # Reset the step that was running so the sidebar doesn't show it as active.
            for k in ["audio", "transcript", "title", "summary", "extract", "rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"X Error: {e}")

# ── Results ──────────────────────────────────────────────────────────────────────
# Shown on every rerun once a result exists (e.g. while chatting).
if st.session_state.result:
    r = st.session_state.result

    # Title banner
    st.markdown(
        f"""
    <div class="card">
        <div class="card-title">{icon("bookmark")} Session Title</div>
        <div style="font-family:'Cormorant Garamond',serif;font-size:1.4rem;font-weight:700;color:var(--text)">
            {esc(r['title'])}
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

    # Top row: summary + transcript
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(
            f"""
        <div class="card">
            <div class="card-title">{icon("list")} Summary</div>
            <div class="card-content">{md(r['summary'])}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        # Interactive transcript: player + clickable, searchable, auto-highlighting
        # lines in one browser-side widget (utils/transcript_view.py).
        st.markdown(
            '<div class="card-title" style="margin-bottom:0.5rem">Interactive transcript '
            "&middot; click a line to jump there</div>",
            unsafe_allow_html=True,
        )
        has_video = bool(youtube_id(r["source"]))
        components.html(
            build_transcript_html(r.get("segments", []), youtube_id(r["source"])),
            height=640 if has_video else 420,
        )

    # Second row: meeting notes. Only sections with content get a card; the
    # empty ones (typical for a tutorial or talk) collapse into one quiet line.
    notes = [
        ("check", "Action Items", "action items", r["action_items"]),
        ("key", "Key Decisions", "key decisions", r["key_decisions"]),
        ("question", "Open Questions", "open questions", r["open_questions"]),
    ]
    filled = [n for n in notes if not nothing_found(n[3])]
    empty = [n[2] for n in notes if nothing_found(n[3])]

    if filled:
        for column, (icon_name, heading, _, text) in zip(st.columns(len(filled), gap="medium"), filled):
            with column:
                st.markdown(
                    f"""
                <div class="card">
                    <div class="card-title">{icon(icon_name)} {heading}</div>
                    <div class="card-content">{md(text)}</div>
                </div>""",
                    unsafe_allow_html=True,
                )

    if empty:
        # "action items, key decisions or open questions"
        listed = empty[0] if len(empty) == 1 else ", ".join(empty[:-1]) + " or " + empty[-1]
        reason = (
            " These appear for meetings, where people assign tasks and make decisions."
            if not filled
            else ""
        )
        st.markdown(
            f'<div class="quiet-note">No {listed} in this recording.{reason}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style=\"font-family:'Cormorant Garamond',serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem;display:flex;align-items:center;gap:0.5rem\">{icon('chat', 20)}<span>Chat with your Meeting</span></div>",
        unsafe_allow_html=True,
    )

    # Chat history display — built as one HTML string so it sits in one scroll box
    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-end">
                    <span class="chat-label user-label">You</span>
                    <div class="chat-bubble user-bubble">{esc(msg['content'])}</div>
                </div>"""
            else:
                # Answer with clickable [mm:ss] citations + the excerpts it used.
                answered = msg["content"].strip() != NOT_FOUND
                body = md(link_citations(msg["content"], r["source"]))
                body += sources_html(msg.get("sources", []), r["source"], answered)
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-start">
                    <span class="chat-label bot-label">Rewi</span>
                    <div class="chat-bubble bot-bubble">{body}</div>
                </div>"""
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
        <div class="card" style="text-align:center;padding:2rem">
            <div style="margin-bottom:0.5rem;color:var(--accent-2-deep)">{icon("chat", 34)}</div>
            <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your meeting transcript</div>
        </div>""",
            unsafe_allow_html=True,
        )

    # Chat input
    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_input = st.text_input(
            "Your question",
            placeholder="What were the main decisions made?",
            label_visibility="collapsed",
        )
    with chat_col2:
        send_btn = st.button("Send", type="primary", use_container_width=True)

    # Ask the RAG chain; on success store both turns and rerun to show them.
    if send_btn and user_input.strip():
        with st.spinner("Thinking…"):
            try:
                result = ask_question(r["rag_chain"], user_input.strip())
            except Exception as error:
                st.error(f"Could not answer your question: {error}")
            else:
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input.strip()}
                )
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        # Keep the retrieved excerpts to show under the answer.
                        "sources": [
                            {"start": d.metadata.get("start", 0), "text": d.page_content}
                            for d in result["sources"]
                        ],
                    }
                )
                st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Empty state
    st.markdown(
        f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3rem 2rem;text-align:center">
        <div style="margin-bottom:0.75rem">{rewi_svg("happy", 130)}</div>
        <div style="font-family:'Cormorant Garamond',serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
            Paste a YouTube link or upload a meeting recording in the sidebar, choose the language, and click <strong>Analyse</strong>.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )
