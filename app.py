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
import re

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
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

/* ── Sidebar: pastel blue panel ── */
[data-testid="stSidebar"] {
    background: var(--accent-2) !important;
    border-right: 1px solid var(--accent-2-deep) !important;
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
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(67, 48, 46, 0.15) !important;
}

.stButton > button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 999px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
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

.stButton > button:hover {
    background: var(--accent-glow) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(67, 48, 46, 0.25) !important;
}

/* Secondary button (Clear Chat): pastel blue with burgundy text */
.stButton > button[kind="secondary"] {
    background: var(--accent-2) !important;
    border: 1px solid var(--accent-2-deep) !important;
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
.chat-bubble a:hover { background: var(--accent-glow); }

/* Retrieved excerpts under each answer */
.sources { margin-top: 0.6rem; border-top: 1px solid var(--accent-2-deep); padding-top: 0.5rem; }
.sources summary { cursor: pointer; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-muted); }
.source { font-size: 0.8rem; line-height: 1.5; margin-top: 0.5rem; padding: 0.5rem 0.65rem; background: var(--surface); border-radius: 10px; color: var(--text); }
.source-time { margin-right: 0.35rem; font-weight: 700; }

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

/* scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--accent-2-deep); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
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
        lambda m: f"[▶ {m.group(1)}]({time_link(source, _to_seconds(m.group(1)))})", answer
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
            f'<a href="{esc(href)}" target="_blank" rel="noopener">▶ {stamp}</a>'
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


def render_step_bar(label: str, key: str, icon: str):
    """Draw one row of the pipeline status list in the sidebar."""
    css = step_status(st.session_state.pipeline_steps, key)
    secs = st.session_state.step_times.get(key)
    timing = f" · {secs:.0f}s" if secs is not None else ""
    st.markdown(
        f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}{timing}</span>
    </div>""",
        unsafe_allow_html=True,
    )


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
    source = st.text_input(
        "YouTube URL or File Path",
        placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4",
    )

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    # Captions are fetched in seconds; Whisper takes about as long as the video.
    use_captions = st.checkbox(
        "Use YouTube captions when available (fast)",
        value=True,
        help="Skips the audio download and Whisper. Untick to always transcribe the audio yourself.",
    )

    run_btn = st.button("⚡  Analyse", type="primary", use_container_width=True)

    # The LLM steps all need this key; fail early with a clear message.
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("GOOGLE_API_KEY is missing from .env — summary, extraction and chat will fail.")

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown(
            '<span class="badge badge-green">Pipeline Status</span>',
            unsafe_allow_html=True,
        )
        for step, icon, label in [
            ("audio", "🔊", "Audio Processing"),
            ("transcript", "📝", "Transcription"),
            ("title", "🏷️", "Title Generation"),
            ("summary", "📋", "Summarisation"),
            ("extract", "🔍", "Extraction"),
            ("rag", "🧠", "RAG Engine"),
        ]:
            render_step_bar(label, step, icon)

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
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        # Clear the previous video's results before starting.
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        st.session_state.step_times = {}

        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            # Live progress box in the main area: one line per stage with its time.
            status = progress_placeholder.status("⚙️ Analysing…", expanded=True)

            def run_stage(label, keys, fn, *args):
                """Mark `keys` active, run fn, mark done, and log how long it took."""
                for k in keys:
                    update_step(k, "active")
                status.update(label=f"⚙️ {label}…")
                start = time.time()
                out = fn(*args)
                secs = time.time() - start
                for k in keys:
                    update_step(k, "done")
                    st.session_state.step_times[k] = secs
                status.write(f"✓ {label} — {secs:.0f}s")
                return out

            # Fast path: English YouTube videos usually have captions with timestamps.
            segments = None
            if use_captions and language == "english" and youtube_id(source):
                segments = run_stage(
                    "Fetching YouTube captions",
                    ["audio", "transcript"],
                    fetch_youtube_captions,
                    source,
                )

            # Fallback (no captions, local file, Hinglish): download + transcribe.
            if not segments:
                chunks = run_stage(
                    "Downloading / converting audio", ["audio"], process_input, source
                )
                # Timestamped segments: plain text for analysis, timings for citations.
                segments = run_stage(
                    "Transcribing", ["transcript"], transcribe_segments, chunks, language
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
            )
            rag_chain = run_stage("Building chat index", ["rag"], build_rag_chain, segments)

            st.session_state.result = {
                **analysis,
                "transcript": transcript,
                "source": source.strip(),  # used to build timestamp links
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            status.update(label="✅ Analysis complete!", state="complete")
            time.sleep(0.5)
            progress_placeholder.empty()
            st.rerun()  # redraw the page so the results section renders

        except Exception as e:
            # Reset the step that was running so the sidebar doesn't show it as active.
            for k in ["audio", "transcript", "title", "summary", "extract", "rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"❌ Error: {e}")

# ── Results ──────────────────────────────────────────────────────────────────────
# Shown on every rerun once a result exists (e.g. while chatting).
if st.session_state.result:
    r = st.session_state.result

    # Title banner
    st.markdown(
        f"""
    <div class="card">
        <div class="card-title">📌 Session Title</div>
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
            <div class="card-title">📋 Summary</div>
            <div class="card-content">{md(r['summary'])}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        # Embedded player for YouTube sources; citations below link to timestamps in it.
        if youtube_id(r["source"]):
            st.video(r["source"])
        with st.expander("📝 Full Transcript", expanded=False):
            st.markdown(
                f'<div class="transcript-box">{esc(r["transcript"])}</div>',
                unsafe_allow_html=True,
            )

    # Second row: action items | decisions | questions
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(
            f"""
        <div class="card">
            <div class="card-title">✅ Action Items</div>
            <div class="card-content">{md(r['action_items'])}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
        <div class="card">
            <div class="card-title">🔑 Key Decisions</div>
            <div class="card-content">{md(r['key_decisions'])}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
        <div class="card">
            <div class="card-title">❓ Open Questions</div>
            <div class="card-content">{md(r['open_questions'])}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown(
        "<div style=\"font-family:'Cormorant Garamond',serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem\">💬 Chat with your Meeting</div>",
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
                    <span class="chat-label bot-label">🤖 Assistant</span>
                    <div class="chat-bubble bot-bubble">{body}</div>
                </div>"""
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown(
            """
        <div class="card" style="text-align:center;padding:2rem">
            <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
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
        send_btn = st.button("Send →", type="primary", use_container_width=True)

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
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Empty state
    st.markdown(
        """
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Cormorant Garamond',serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
            Paste a YouTube URL or local file path in the sidebar, choose your language, and hit <strong>Analyse</strong> to get started.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )
