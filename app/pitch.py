"""Football pitch with both lineups, drawn in Altair so a click on a player can be read back
(st.altair_chart on_select). Vertical, so it reads on a phone: home at the bottom attacking
up, away at the top attacking down.

API-Football grid 'row:col': row 1 is the goalkeeper; col 1 is the team's left flank. Facing
up, a team's left is the left of the screen; facing down, it is the right.
"""

from __future__ import annotations

import base64
import io

import altair as alt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

LENGTH, WIDTH = 105.0, 68.0
GRASS, LINE = "#3f8f4f", "#e9f3ea"
TEAM_COLORS = {"home": "#2a78d6", "away": "#eb6834"}
DOUBTFUL = "#eda100"
PHOTO_PX = 96


# --------------------------------------------------------------------------- coordinates


def row_y(row: int, formation: str | None, side: str) -> float:
    """Distance along the pitch: goalkeeper near his goal, last line just short of halfway."""
    rows = len(formation.split("-")) + 1 if formation else 5
    y = 5.0 + (row - 1) * (43.0 / max(rows - 1, 1))
    return y if side == "home" else LENGTH - y


def place(players: pd.DataFrame) -> pd.DataFrame:
    """Add x, y to rows with grid, formation and side ('home' / 'away')."""
    p = players.copy()
    parts = p.grid.str.split(":", expand=True).astype(int)
    p["row"], p["col"] = parts[0], parts[1]
    p["n_in_row"] = p.groupby(["side", "row"]).col.transform("max")
    p["y"] = [row_y(r, f, s) for r, f, s in zip(p.row, p.formation, p.side, strict=True)]
    frac = (p.col - 0.5) / p.n_in_row
    p["x"] = np.where(p.side == "home", WIDTH * frac, WIDTH * (1 - frac))
    return p


# --------------------------------------------------------------------------- photos


def _circle(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA").resize((PHOTO_PX, PHOTO_PX))
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse((0, 0, PHOTO_PX - 1, PHOTO_PX - 1), fill=255)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def initials_avatar(name: str, color: str) -> Image.Image:
    img = Image.new("RGBA", (PHOTO_PX, PHOTO_PX), color)
    letters = "".join(w[0] for w in name.replace(".", " ").split()[:2]).upper() or "?"
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=36)
    box = draw.textbbox((0, 0), letters, font=font)
    pos = ((PHOTO_PX - (box[2] - box[0])) / 2, (PHOTO_PX - (box[3] - box[1])) / 2 - box[1])
    draw.text(pos, letters, fill="white", font=font)
    return img


def photo_uri(photo: bytes | None, name: str, color: str) -> str:
    """Round PNG data URI (small: ~6 KB) of the photo, or of the initials if there is none."""
    try:
        img = Image.open(io.BytesIO(photo)) if photo else initials_avatar(name, color)
    except OSError:
        img = initials_avatar(name, color)
    buf = io.BytesIO()
    _circle(img).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- chart


def _lines() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pitch markings as segments (x, y, x2, y2) and the centre circle as a polyline."""
    box_w, box_d, six_w, six_d = 40.3, 16.5, 18.3, 5.5
    c = WIDTH / 2
    # Drawn along x = length, then transposed: the pitch is vertical.
    seg = [(0, 0, LENGTH, 0), (0, WIDTH, LENGTH, WIDTH), (0, 0, 0, WIDTH),
           (LENGTH, 0, LENGTH, WIDTH), (LENGTH / 2, 0, LENGTH / 2, WIDTH)]  # fmt: skip
    for x0, d in ((0, 1), (LENGTH, -1)):
        for w, depth in ((box_w, box_d), (six_w, six_d)):
            seg += [(x0, c - w / 2, x0 + d * depth, c - w / 2),
                    (x0, c + w / 2, x0 + d * depth, c + w / 2),
                    (x0 + d * depth, c - w / 2, x0 + d * depth, c + w / 2)]  # fmt: skip
    segments = pd.DataFrame(seg, columns=["y", "x", "y2", "x2"])
    t = np.linspace(0, 2 * np.pi, 60)
    circle = pd.DataFrame({"y": LENGTH / 2 + 9.15 * np.cos(t), "x": c + 9.15 * np.sin(t),
                           "i": range(60)})  # fmt: skip
    return segments, circle


def pitch_chart(players: pd.DataFrame, height: int = 720) -> alt.LayerChart:
    """players: x, y, side, player_id, label, photo, tooltip columns, doubtful."""
    xs = alt.Scale(domain=[-2, WIDTH + 2], nice=False, padding=0)
    ys = alt.Scale(domain=[-4, LENGTH + 2], nice=False, padding=0)
    x = alt.X("x:Q", scale=xs, axis=None)
    y = alt.Y("y:Q", scale=ys, axis=None)
    segments, circle = _lines()

    field = (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0], "x2": [WIDTH], "y2": [LENGTH]}))
        .mark_rect(color=GRASS)
        .encode(x=x, y=y, x2="x2", y2="y2")
    )
    marks = (
        alt.Chart(segments)
        .mark_rule(color=LINE, strokeWidth=1.5)
        .encode(x=x, y=y, x2="x2", y2="y2")
    )
    centre = alt.Chart(circle).mark_line(color=LINE, strokeWidth=1.5).encode(x=x, y=y, order="i")

    pick = alt.selection_point(name="player", fields=["player_id"], on="click", empty=False)
    ring_color = alt.condition(
        "datum.doubtful", alt.value(DOUBTFUL),
        alt.Color("side:N", scale=alt.Scale(domain=list(TEAM_COLORS), range=list(TEAM_COLORS.values())),
                  legend=None),
    )  # fmt: skip
    base = alt.Chart(players).encode(x=x, y=y)
    rings = base.mark_circle(size=1700, opacity=1).encode(color=ring_color)
    tooltip = [c for c in players.columns if c.startswith("tt_")]
    photos = base.mark_image(width=36, height=36).encode(url="photo:N")
    # Clicks and tooltips go to an (almost) invisible circle on top of each photo: Vega does not
    # pick image marks reliably, circles it does.
    hit = (
        base.mark_circle(size=1700, opacity=0.001, cursor="pointer")
        .encode(tooltip=[alt.Tooltip(c, title=c[3:]) for c in tooltip])
        .add_params(pick)
    )
    names = base.mark_text(dy=28, fontSize=10, fontWeight="bold", color="white").encode(
        text="label:N"
    )
    return (
        alt.layer(field, marks, centre, rings, photos, names, hit)
        .properties(height=height, width="container")
        .configure_view(stroke=None, fill="#2f6f3d")
    )
