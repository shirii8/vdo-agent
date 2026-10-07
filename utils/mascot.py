"""
Rewi, the app's mascot: a note page with a play-button antenna.

`rewi_svg(mood, size)` returns inline SVG for one of four expressions:
  happy     - default smile (start screen, logo)
  listening - round mouth and sound waves (fetching / transcribing)
  thinking  - flat mouth and three bouncing dots (writing the summary)
  done      - wink, big smile and a check badge (finished)

The SVG carries only shapes and class names. Colours and animations come from
the stylesheet in app.py (the "Rewi" block), using the palette variables
--accent, --bg and --accent-2, so the mascot always matches the theme.
"""

# Body parts shared by every mood. Legs sit outside the bobbing group and are
# tall, so no gap shows while the body moves up and down.
_LEGS = (
    '<rect class="rw-b" x="80" y="164" width="18" height="42" rx="6"/>'
    '<rect class="rw-b" x="142" y="164" width="18" height="42" rx="6"/>'
    '<rect class="rw-b" x="64" y="196" width="36" height="14" rx="7"/>'
    '<rect class="rw-b" x="140" y="196" width="36" height="14" rx="7"/>'
)
_ANTENNA = (
    '<g class="rw-antenna">'
    '<path class="rw-bs" d="M80 42 L70 20" stroke-width="7" stroke-linecap="round"/>'
    '<path class="rw-b rw-bs" d="M60 8 L80 18 L60 28 Z" stroke-width="5" stroke-linejoin="round"/>'
    "</g>"
)
_BODY = (
    '<path class="rw-b" d="M64 40 H166 L200 74 V156 Q200 180 176 180 H64 Q40 180 40 156 V64 Q40 40 64 40 Z"/>'
    '<path class="rw-a" d="M166 40 V62 Q166 74 178 74 H200 Z"/>'
)
_CHEEKS = (
    '<ellipse class="rw-a rw-cheek" cx="80" cy="124" rx="10" ry="6"/>'
    '<ellipse class="rw-a rw-cheek" cx="160" cy="124" rx="10" ry="6"/>'
)
_EYES = (
    '<circle class="rw-e rw-eye" cx="98" cy="102" r="10"/>'
    '<circle class="rw-e rw-eye" cx="142" cy="102" r="10"/>'
)

# Per-mood face, plus extras drawn around the body.
_FACES = {
    "happy": _EYES
    + '<path class="rw-es" d="M96 130 Q120 152 144 130" stroke-width="10" stroke-linecap="round"/>',
    "listening": _EYES + '<circle class="rw-e" cx="120" cy="138" r="8"/>',
    "thinking": _EYES
    + '<path class="rw-es" d="M106 138 H134" stroke-width="9" stroke-linecap="round"/>',
    "done": '<circle class="rw-e rw-eye" cx="98" cy="102" r="10"/>'
    # winking right eye
    + '<path class="rw-es" d="M131 106 Q142 94 153 106" stroke-width="8" stroke-linecap="round"/>'
    + '<path class="rw-es" d="M92 128 Q120 158 148 128" stroke-width="10" stroke-linecap="round"/>',
}
_EXTRAS = {
    "happy": "",
    # sound waves on both sides
    "listening": (
        '<g class="rw-bs" stroke-width="6" stroke-linecap="round">'
        '<path class="rw-wave" d="M28 94 Q18 110 28 126"/>'
        '<path class="rw-wave rw-wave-2" d="M14 84 Q-2 110 14 136"/>'
        '<path class="rw-wave" d="M212 94 Q222 110 212 126"/>'
        '<path class="rw-wave rw-wave-2" d="M226 84 Q242 110 226 136"/>'
        "</g>"
    ),
    # three dots above the right corner
    "thinking": (
        '<circle class="rw-b rw-dot" cx="176" cy="14" r="6"/>'
        '<circle class="rw-b rw-dot rw-dot-2" cx="196" cy="6" r="6"/>'
        '<circle class="rw-b rw-dot rw-dot-3" cx="216" cy="14" r="6"/>'
    ),
    # check badge on the bottom-right corner
    "done": (
        '<g class="rw-badge">'
        '<circle class="rw-ring" cx="198" cy="172" r="22" stroke-width="7"/>'
        '<path class="rw-bs" d="M187 172 L195 180 L210 164" stroke-width="7" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        "</g>"
    ),
}


def rewi_svg(mood: str = "happy", size: int = 120, animated: bool = True) -> str:
    """Inline SVG of Rewi. `animated=False` gives a still version (for the logo)."""
    mood = mood if mood in _FACES else "happy"
    still = "" if animated else " rw-still"
    height = round(size * 252 / 240)
    return (
        f'<svg class="rewi rw-{mood}{still}" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="-8 -12 256 252" width="{size}" height="{height}" role="img" aria-label="Rewi">'
        f"{_LEGS}"
        f'<g class="rw-bob">{_ANTENNA}{_BODY}{_CHEEKS}{_FACES[mood]}</g>'
        f"{_EXTRAS[mood]}"
        "</svg>"
    )
