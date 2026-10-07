"""
Interactive transcript: a self-contained HTML/JS widget that app.py embeds
with st.components.v1.html.

  - YouTube player and transcript side by side in one widget
  - click a sentence -> the video seeks to that moment
  - the sentence being spoken is highlighted and kept in view
  - search box highlights matches; Enter / Shift+Enter step through them

Everything runs in the browser: no Streamlit reruns and no model calls, so it
adds nothing to the pipeline time. For local files (no YouTube id) the player
is left out and the transcript is still searchable.
"""

import json

# A timestamp label starts a new paragraph roughly every this many seconds.
PARAGRAPH_SECONDS = 30

_TEMPLATE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --surface: #FFFDF6; --surface-2: #FFF1B5; --border: #E8D9A0;
    --accent: #43302E; --blue: #C1DBE8; --blue-deep: #8DB8CE; --muted: #7A6461;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font-family: 'DM Sans', sans-serif; color: var(--accent); background: transparent; }
  .wrap { display: flex; flex-direction: column; height: 100%; gap: 10px; }
  .player { width: 100%; aspect-ratio: 16 / 9; border-radius: 14px; overflow: hidden; background: #000; flex-shrink: 0; }
  .player iframe { width: 100%; height: 100%; display: block; }
  .bar { display: flex; gap: 8px; align-items: center; flex-shrink: 0; }
  .bar input[type=search] {
    flex: 1; min-width: 0; padding: 7px 12px; border: 1px solid var(--border); border-radius: 999px;
    background: var(--surface); color: var(--accent); font: inherit; font-size: 13px; outline: none;
  }
  .bar input[type=search]:focus { border-color: var(--accent); }
  .count { font-size: 12px; color: var(--muted); white-space: nowrap; min-width: 54px; text-align: right; }
  .follow { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 4px; white-space: nowrap; cursor: pointer; }
  .follow input { accent-color: var(--accent); }
  .lines {
    flex: 1; min-height: 0; overflow-y: auto; padding: 12px 14px; background: var(--surface);
    border: 1px solid var(--border); border-radius: 14px; font-size: 13.5px; line-height: 1.75;
  }
  .para { margin: 0 0 10px 0; }
  .stamp {
    display: inline-block; margin-right: 8px; padding: 0 8px; border-radius: 999px; border: none;
    background: var(--blue); color: var(--accent); font: inherit; font-size: 11px; font-weight: 700; line-height: 20px;
  }
  .seg { border-radius: 4px; padding: 1px 0; transition: background-color 0.15s; }
  .seekable .seg, .seekable .stamp { cursor: pointer; }
  .seekable .seg:hover { background: var(--blue); }
  .seekable .stamp:hover { background: var(--blue-deep); }
  .seg.now { background: var(--surface-2); box-shadow: 0 0 0 2px var(--surface-2); font-weight: 500; }
  mark { background: var(--accent); color: var(--surface-2); border-radius: 3px; padding: 0 1px; }
  .seg.hit-current mark { outline: 2px solid var(--blue-deep); }
  .empty { color: var(--muted); font-size: 13px; }
  /* scrollbar hidden; the box still scrolls with the wheel or touch */
  .lines { scrollbar-width: none; }
  .lines::-webkit-scrollbar { display: none; }
</style>
</head>
<body>
<div class="wrap">
  <div class="player" id="playerBox"><div id="player"></div></div>
  <div class="bar">
    <input type="search" id="q" placeholder="Search the transcript" autocomplete="off">
    <span class="count" id="count"></span>
    <label class="follow" id="followBox" title="Keep the line being spoken in view while the video plays"><input type="checkbox" id="follow" checked> Auto-scroll</label>
  </div>
  <div class="lines" id="lines"></div>
</div>

<script>
// [start seconds, end seconds, text] for every transcript segment.
const SEGS = __SEGMENTS__;
const VIDEO_ID = __VIDEO_ID__;
const PARA_SECONDS = __PARA_SECONDS__;

const linesEl = document.getElementById('lines');
const qEl = document.getElementById('q');
const countEl = document.getElementById('count');
const followEl = document.getElementById('follow');

function stamp(t) {
  t = Math.floor(t);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  const mm = String(m).padStart(2, '0'), ss = String(s).padStart(2, '0');
  return h ? h + ':' + mm + ':' + ss : mm + ':' + ss;
}
function esc(s) {
  return s.replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
}

// ---- Build the transcript: one <span> per segment, grouped into paragraphs ----
const spans = [];
let para = null, paraStart = -1e9;
SEGS.forEach((seg, i) => {
  if (!para || seg[0] - paraStart >= PARA_SECONDS) {
    para = document.createElement('p');
    para.className = 'para';
    const b = document.createElement('button');
    b.className = 'stamp';
    b.textContent = stamp(seg[0]);
    b.dataset.i = i;
    para.appendChild(b);
    linesEl.appendChild(para);
    paraStart = seg[0];
  }
  const sp = document.createElement('span');
  sp.className = 'seg';
  sp.dataset.i = i;
  sp.textContent = seg[2] + ' ';
  para.appendChild(sp);
  spans.push(sp);
});
if (!SEGS.length) linesEl.innerHTML = '<div class="empty">No transcript available.</div>';

// ---- YouTube player (IFrame API). Without a video id the player is hidden. ----
let player = null, ready = false;
if (VIDEO_ID) {
  linesEl.classList.add('seekable');
  const tag = document.createElement('script');
  tag.src = 'https://www.youtube.com/iframe_api';
  document.head.appendChild(tag);
  window.onYouTubeIframeAPIReady = () => {
    player = new YT.Player('player', {
      videoId: VIDEO_ID,
      playerVars: {playsinline: 1, rel: 0, modestbranding: 1},
      events: {onReady: () => { ready = true; }},
    });
  };
} else {
  document.getElementById('playerBox').style.display = 'none';
  document.getElementById('followBox').style.display = 'none';
}

function seek(i) {
  if (!VIDEO_ID) return;
  const t = SEGS[i][0];
  if (ready && player.seekTo) {
    player.seekTo(t, true);
    player.playVideo();
    followEl.checked = true;
    setNow(i);
  } else {
    // Player blocked or not loaded yet: open YouTube at that moment instead.
    window.open('https://www.youtube.com/watch?v=' + VIDEO_ID + '&t=' + Math.floor(t) + 's', '_blank');
  }
}
linesEl.addEventListener('click', e => {
  const el = e.target.closest('[data-i]');
  if (el) seek(Number(el.dataset.i));
});

// ---- Highlight the segment being spoken ----
let now = -1;
function scrollTo(el) {
  linesEl.scrollTo({top: el.offsetTop - linesEl.offsetTop - linesEl.clientHeight / 3, behavior: 'smooth'});
}
function setNow(i) {
  if (i === now) return;
  if (now >= 0) spans[now].classList.remove('now');
  now = i;
  if (i >= 0) {
    spans[i].classList.add('now');
    if (followEl.checked) scrollTo(spans[i]);
  }
}
function indexAt(t) {
  // Last segment that starts at or before t (binary search).
  let lo = 0, hi = SEGS.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (SEGS[mid][0] <= t) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return ans;
}
setInterval(() => {
  if (!ready || !player.getCurrentTime || !player.getPlayerState) return;
  if (player.getPlayerState() !== 1) return;  // 1 = playing
  setNow(indexAt(player.getCurrentTime()));
}, 400);
// Scrolling by hand means "let me read": stop following until re-ticked or a line is clicked.
['wheel', 'touchmove'].forEach(ev => linesEl.addEventListener(ev, () => { followEl.checked = false; }, {passive: true}));

// ---- Search ----
let hits = [], hitPos = -1;
function runSearch() {
  const q = qEl.value.trim().toLowerCase();
  hits = []; hitPos = -1;
  spans.forEach((sp, i) => {
    const text = SEGS[i][2] + ' ';
    sp.classList.remove('hit-current');
    if (!q) { if (sp.firstElementChild) sp.textContent = text; return; }
    const lower = text.toLowerCase();
    let from = 0, at = lower.indexOf(q), out = '';
    if (at < 0) { if (sp.firstElementChild) sp.textContent = text; return; }
    while (at >= 0) {
      out += esc(text.slice(from, at)) + '<mark>' + esc(text.slice(at, at + q.length)) + '</mark>';
      from = at + q.length;
      at = lower.indexOf(q, from);
    }
    sp.innerHTML = out + esc(text.slice(from));
    hits.push(i);
  });
  if (!q) { countEl.textContent = ''; return; }
  if (!hits.length) { countEl.textContent = 'No matches'; return; }
  step(1);
}
function step(dir) {
  if (!hits.length) return;
  if (hitPos >= 0) spans[hits[hitPos]].classList.remove('hit-current');
  hitPos = (hitPos + dir + hits.length) % hits.length;
  const sp = spans[hits[hitPos]];
  sp.classList.add('hit-current');
  followEl.checked = false;
  scrollTo(sp);
  countEl.textContent = (hitPos + 1) + ' of ' + hits.length;
}
let timer = null;
qEl.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(runSearch, 150); });
qEl.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); step(e.shiftKey ? -1 : 1); } });
</script>
</body>
</html>
"""


def _js(value) -> str:
    """JSON for embedding inside a <script> tag (a literal '</' would end the tag)."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def build_transcript_html(segments: list, video_id: str | None = None) -> str:
    """Return the widget's HTML for a list of {"start", "end", "text"} segments."""
    rows = [
        [round(float(s["start"]), 2), round(float(s["end"]), 2), s["text"]]
        for s in segments
        if s.get("text")
    ]
    # Segments go in last so transcript text can never collide with a placeholder.
    return (
        _TEMPLATE.replace("__VIDEO_ID__", _js(video_id))
        .replace("__PARA_SECONDS__", str(PARAGRAPH_SECONDS))
        .replace("__SEGMENTS__", _js(rows))
    )
