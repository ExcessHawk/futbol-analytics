"""Table colouring rules shared by every view (pandas Styler -> st.dataframe).

Each colour has one job, so the same colour always means the same thing:
  * status    win / draw / loss and table zones: fixed status colours, always paired
              with a letter or a legend so colour is never the only cue
  * diverging signed differences (goal difference, points vs expected, goals - xG):
              blue = above, red = below, grey midpoint; intensity = size of the gap
  * sequential magnitudes (shots, xG, per-90 rates): one blue hue, darker = more
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

# Validated palette (light surface).
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
NEUTRAL = "#c9c8c2"
BLUE, RED = "#2a78d6", "#e34948"
MIDPOINT = "#f0efec"
SURFACE = "#fcfcfb"
SEQ_DARK = "#1c5cab"
INK, WHITE = "#0b0b0b", "#ffffff"

FORM_COLORS = {"G": (GOOD, WHITE), "E": (NEUTRAL, INK), "P": (CRITICAL, WHITE)}
FORM_COLUMNS = [f"form_{i}" for i in range(1, 6)]


# --------------------------------------------------------------------------- colour maths


def _rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)


def _mix(a: str, b: str, t: float) -> str:
    """Linear blend from colour a (t=0) to colour b (t=1)."""
    c = _rgb(a) + (_rgb(b) - _rgb(a)) * float(np.clip(t, 0, 1))
    return "#" + "".join(f"{int(round(v)):02x}" for v in c)


def _cell(bg: str, fg: str, bold: bool = False) -> str:
    return f"background-color: {bg}; color: {fg};" + (" font-weight: 700;" if bold else "")


def _ink_for(t: float) -> str:
    """Dark text on light fills, white on dark fills (keeps contrast >= 4.5:1)."""
    return WHITE if t > 0.55 else INK


# --------------------------------------------------------------------------- column rules


def diverging(col: pd.Series, cap: float | None = None) -> list[str]:
    """Blue for positive, red for negative; intensity relative to the largest |value|."""
    limit = cap or float(np.nanmax(np.abs(col.to_numpy(float)))) or 1.0
    out = []
    for v in col:
        if pd.isna(v):
            out.append("")
            continue
        t = min(abs(float(v)) / limit, 1.0) * 0.85  # never fully saturated: text stays readable
        out.append(_cell(_mix(MIDPOINT, BLUE if v >= 0 else RED, t), _ink_for(t)))
    return out


def sequential(col: pd.Series, reverse: bool = False) -> list[str]:
    """Light -> dark blue by value within the column (reverse when lower is 'more')."""
    values = col.to_numpy(float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    span = (hi - lo) or 1.0
    out = []
    for v in values:
        if np.isnan(v):
            out.append("")
            continue
        t = (v - lo) / span
        t = (1 - t) if reverse else t
        out.append(_cell(_mix(SURFACE, SEQ_DARK, t * 0.8), _ink_for(t * 0.8)))
    return out


def form_cell(letter: str) -> str:
    if letter not in FORM_COLORS:
        return ""
    bg, fg = FORM_COLORS[letter]
    return _cell(bg, fg, bold=True) + " text-align: center;"


# --------------------------------------------------------------------------- table zones

# Final-table zones for the 2026/27 season rules. Champions League: top 4 in all five
# leagues (an extra place can be earned through UEFA coefficients; not modelled).
# Bundesliga and Ligue 1 (18 teams) send 16th to a relegation play-off.
ZONES = {
    "E0": {"champions": 4, "playoff": None, "relegated": 3},
    "SP1": {"champions": 4, "playoff": None, "relegated": 3},
    "I1": {"champions": 4, "playoff": None, "relegated": 3},
    "D1": {"champions": 4, "playoff": 1, "relegated": 2},
    "F1": {"champions": 4, "playoff": 1, "relegated": 2},
}
# 2015/16 rules, for the StatsBomb season: Serie A and Ligue 1 had only 3 Champions League
# places, and Ligue 1 still had 20 teams with 3 relegated directly.
ZONES_2015_16 = {
    "E0": {"champions": 4, "playoff": None, "relegated": 3},
    "SP1": {"champions": 4, "playoff": None, "relegated": 3},
    "I1": {"champions": 3, "playoff": None, "relegated": 3},
    "F1": {"champions": 3, "playoff": None, "relegated": 3},
}
ZONE_LEGEND = {
    "champions": (BLUE, "Champions League"),
    "playoff": (WARNING, "Playoff de descenso"),
    "relegated": (CRITICAL, "Descenso"),
}


def zone_of(position: int, n_teams: int, league: str, rules: dict | None = None) -> str | None:
    z = (rules or ZONES).get(league)
    if not z:
        return None
    if position <= z["champions"]:
        return "champions"
    if position > n_teams - z["relegated"]:
        return "relegated"
    if z["playoff"] and position > n_teams - z["relegated"] - z["playoff"]:
        return "playoff"
    return None


def zone_cells(positions: pd.Series, league: str, rules: dict | None = None) -> list[str]:
    n = len(positions)
    out = []
    for p in positions:
        zone = zone_of(int(p), n, league, rules)
        if zone is None:
            out.append("")
            continue
        bg = ZONE_LEGEND[zone][0]
        out.append(
            _cell(bg, INK if zone == "playoff" else WHITE, bold=True) + " text-align: center;"
        )
    return out


def zone_legend_html(league: str, rules: dict | None = None, show_form: bool = True) -> str:
    """Small legend chips under the table: zone colours always come with their label."""
    z = (rules or ZONES).get(league, {})
    chips = []
    for key, (color, label) in ZONE_LEGEND.items():
        if z.get(key):
            chips.append(
                f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px">'
                f'<span style="width:12px;height:12px;border-radius:3px;background:{color}"></span>{label}</span>'
            )
    if show_form:
        chips.append(
            '<span style="margin-left:8px">Forma: '
            + "".join(
                f'<span style="display:inline-block;width:20px;text-align:center;margin:0 2px;border-radius:3px;'
                f'font-weight:700;background:{bg};color:{fg}">{k}</span>'
                for k, (bg, fg) in FORM_COLORS.items()
            )
            + " victoria · empate · derrota</span>"
        )
    return f'<div style="font-size:0.85rem;color:#52514e;margin-top:4px">{"".join(chips)}</div>'


# --------------------------------------------------------------------------- helpers


def split_form(form: pd.Series) -> pd.DataFrame:
    """'G E P G G' -> five columns, oldest first, padded on the left for short seasons."""
    rows = []
    for f in form:
        letters = f.split() if isinstance(f, str) else []
        rows.append([""] * (5 - len(letters)) + letters[-5:])
    return pd.DataFrame(rows, columns=FORM_COLUMNS, index=form.index)


def apply_columns(styler: Styler, rule, columns: Iterable[str], **kwargs) -> Styler:
    """Apply a column rule (diverging / sequential) to each existing column separately."""
    for c in columns:
        if c in styler.data.columns:
            styler = styler.apply(lambda s, _r=rule, _k=kwargs: _r(s, **_k), subset=[c], axis=0)
    return styler
