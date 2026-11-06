"""A synthetic bet slip, for the dashboard demo and for end-to-end OCR tests.

Generic layout (no real bookmaker's branding): per pick the selection and its odds
come first, then the market, then the match and kick-off - the "details above the
match" layout. One price is American (+110) and team names are the long ones a slip
uses ("Nottingham Forest"), which is what makes the parser earn its keep.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SAMPLE_PICKS = [
    ("Arsenal", "1.40", "Resultado final", "Arsenal vs Hull City", "sáb 7 nov · 09:00"),
    ("Más de 2.5", "1.85", "Total de goles", "Fulham vs Newcastle United", "sáb 7 nov · 09:00"),
    (
        "Sí",
        "1.75",
        "Ambos equipos anotan",
        "Nottingham Forest vs Manchester City",
        "sáb 7 nov · 11:30",
    ),
    ("Chelsea", "+110", "Resultado final", "Sunderland vs Chelsea", "dom 8 nov · 08:00"),
    (
        "Más de 9.5",
        "1.90",
        "Total de córners",
        "Leeds United vs Tottenham Hotspur",
        "sáb 7 nov · 06:30",
    ),
    # A player prop: needs the player's season average (API-Football or typed by hand).
    (
        "Kylian Mbappé",
        "2.40",
        "Tiros a puerta: 3 o más",
        "Valencia vs Real Madrid",
        "sáb 7 nov · 14:00",
    ),
]

STAKE = 100.0


def decimal_odds(text: str) -> float:
    """'1.85' -> 1.85, '+110' -> 2.10, '-250' -> 1.40."""
    if text[0] in "+-":
        v = int(text)
        return 1 + v / 100 if v > 0 else 1 + 100 / abs(v)
    return float(text)


def sample_total_odds() -> float:
    """The slip's combined odds: the product of every pick's (decimal) odds."""
    return float(np.prod([decimal_odds(p[1]) for p in SAMPLE_PICKS]))


FONT_CANDIDATES = {
    "regular": ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "bold": [
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
}


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES[kind]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def render_ticket(dark: bool = False) -> Image.Image:
    """Phone-screenshot-like slip, 720 px wide."""
    bg, fg, muted, accent = (
        ("#16181d", "#f2f2f2", "#9aa0a6", "#3987e5")
        if dark
        else ("#ffffff", "#111111", "#666666", "#2a78d6")
    )
    w, row_h = 720, 150
    h = 130 + row_h * len(SAMPLE_PICKS) + 170
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    d.text(
        (32, 36), f"Mi boleto · Combinada ({len(SAMPLE_PICKS)})", font=_font("bold", 34), fill=fg
    )
    y = 120
    for selection, odds, market, match, when in SAMPLE_PICKS:
        d.line((32, y - 14, w - 32, y - 14), fill=muted, width=1)
        d.text((32, y), selection, font=_font("bold", 30), fill=fg)
        tw = d.textlength(odds, font=_font("bold", 30))
        d.text((w - 32 - tw, y), odds, font=_font("bold", 30), fill=accent)
        d.text((32, y + 42), market, font=_font("regular", 24), fill=muted)
        d.text((32, y + 76), match, font=_font("regular", 26), fill=fg)
        d.text((32, y + 108), when, font=_font("regular", 20), fill=muted)
        y += row_h
    d.line((32, y - 14, w - 32, y - 14), fill=muted, width=1)
    total = sample_total_odds()
    d.text((32, y + 10), f"Cuota total {total:.2f}", font=_font("bold", 28), fill=fg)
    d.text((32, y + 56), f"Apuesta ${STAKE:.2f}", font=_font("regular", 26), fill=fg)
    d.text(
        (32, y + 98),
        f"Ganancia potencial ${STAKE * total:,.2f}",
        font=_font("regular", 26),
        fill=fg,
    )
    return img


def as_phone_photo(img: Image.Image, angle: float = 3.0, seed: int = 0) -> Image.Image:
    """Degrade a clean slip like a phone photo: tilt, blur, uneven light, noise, JPEG."""
    rng = np.random.default_rng(seed)
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="#d9d6cf")
    blurred = rotated.filter(ImageFilter.GaussianBlur(1.1))
    arr = np.asarray(blurred).astype(float)
    h, w = arr.shape[:2]
    shade = np.linspace(1.0, 0.72, w)[None, :, None] * np.linspace(1.0, 0.9, h)[:, None, None]
    arr = np.clip(arr * shade + rng.normal(0, 6, arr.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=70)
    return Image.open(io.BytesIO(buf.getvalue()))


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
