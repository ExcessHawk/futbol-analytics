"""Pitch visualisations with mplsoccer (Phase 4).

Every function takes plain DataFrames and returns a matplotlib Figure, so the same code
serves notebooks, scripts (fig.savefig) and Streamlit (st.pyplot).

Colour rules: categorical hues in fixed order (blue = primary series, orange = second,
aqua = third), one-hue blue ramp for magnitudes, text in neutral ink, never series colour.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from mplsoccer import Pitch, Radar, VerticalPitch
from scipy.ndimage import gaussian_filter

# Validated categorical palette (light surface), fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK_MUTED = "#0b0b0b", "#52514e"
LINE = "#b5b4ae"
NEUTRAL_MARK = "#c9c8c2"
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK})


def _title(fig: Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.02, 0.97, title, fontsize=14, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.02, 0.915, subtitle, fontsize=10, color=INK_MUTED, va="top")


def _pitch_kwargs() -> dict:
    return dict(pitch_type="statsbomb", pitch_color=SURFACE, line_color=LINE, linewidth=1)


def shot_map(shots: pd.DataFrame, title: str, subtitle: str | None = None) -> Figure:
    """Attacking half, marker area proportional to xG, goals highlighted.

    shots columns: x, y, shot_xg, outcome ('Goal' marks a goal)
    """
    pitch = VerticalPitch(half=True, pad_bottom=0.5, **_pitch_kwargs())
    fig, ax = pitch.draw(figsize=(7, 6.6))
    fig.set_facecolor(SURFACE)
    goals = shots[shots.outcome == "Goal"]
    misses = shots[shots.outcome != "Goal"]
    size = lambda xg: 40 + 900 * xg  # noqa: E731 - area, so 0.05 xG vs 0.5 xG is 10x
    pitch.scatter(misses.x, misses.y, s=size(misses.shot_xg), color=BLUE, alpha=0.45,
                  edgecolors=SURFACE, linewidth=1.5, ax=ax, label="Tiro")  # fmt: skip
    pitch.scatter(goals.x, goals.y, s=size(goals.shot_xg), color=ORANGE,
                  edgecolors=SURFACE, linewidth=1.5, ax=ax, label="Gol", zorder=3)  # fmt: skip
    ax.legend(loc="lower left", frameon=False, labelcolor=INK_MUTED, fontsize=9, markerscale=0.6)
    summary = f"{len(shots)} tiros · {shots.shot_xg.sum():.1f} xG · {len(goals)} goles"
    _title(fig, title, subtitle or f"{summary} · tamaño = xG del tiro")
    return fig


def pass_map(
    passes: pd.DataFrame, progressive: pd.Series, title: str, subtitle: str | None = None
) -> Figure:
    """Completed passes as arrows; progressive ones in colour, the rest recessive grey.

    passes columns: x, y, end_x, end_y. `progressive` is a boolean mask aligned to passes.
    """
    pitch = Pitch(**_pitch_kwargs())
    fig, ax = pitch.draw(figsize=(9, 6.6))
    fig.set_facecolor(SURFACE)
    rest, prog = passes[~progressive], passes[progressive]
    pitch.arrows(rest.x, rest.y, rest.end_x, rest.end_y, width=1, headwidth=4, headlength=4,
                 color=NEUTRAL_MARK, alpha=0.6, ax=ax, label="Pase")  # fmt: skip
    pitch.arrows(prog.x, prog.y, prog.end_x, prog.end_y, width=1.6, headwidth=5, headlength=5,
                 color=BLUE, ax=ax, label="Pase progresivo", zorder=3)  # fmt: skip
    ax.legend(loc="lower left", frameon=False, labelcolor=INK_MUTED, fontsize=9,
              bbox_to_anchor=(0, -0.08), ncol=2)  # fmt: skip
    ax.annotate("", xy=(75, -3), xytext=(45, -3), annotation_clip=False,
                arrowprops=dict(arrowstyle="->", color=INK_MUTED))  # fmt: skip
    ax.text(60, -5, "Dirección de ataque", ha="center", color=INK_MUTED, fontsize=8)
    _title(fig, title, subtitle or f"{len(passes)} pases completados · {len(prog)} progresivos")
    return fig


def heatmap(actions: pd.DataFrame, title: str, subtitle: str | None = None) -> Figure:
    """Smoothed density of where a player performs on-ball actions.

    Binned 2D histogram + Gaussian blur instead of a KDE: no seaborn dependency and
    the bin grid matches the pitch dimensions.
    """
    pitch = Pitch(line_zorder=2, **_pitch_kwargs())
    fig, ax = pitch.draw(figsize=(9, 6.6))
    fig.set_facecolor(SURFACE)
    stats = pitch.bin_statistic(actions.x, actions.y, statistic="count", bins=(30, 20))
    stats["statistic"] = gaussian_filter(stats["statistic"], 1.2)
    pitch.heatmap(stats, ax=ax, cmap=SEQUENTIAL_BLUE, edgecolors=None)
    _title(fig, title, subtitle or f"{len(actions)} acciones con balón · ataca hacia la derecha")
    return fig


def radar(
    params: list[str],
    values_a: list[float],
    label_a: str,
    low: list[float],
    high: list[float],
    values_b: list[float] | None = None,
    label_b: str | None = None,
    title: str = "",
    subtitle: str | None = None,
) -> Figure:
    """Per-90 comparison radar. low/high are the axis bounds (5th/95th percentiles)."""
    r = Radar(
        params,
        low,
        high,
        round_int=[False] * len(params),
        num_rings=4,
        ring_width=1,
        center_circle_radius=1,
    )
    fig, ax = plt.subplots(figsize=(8, 8.4))
    fig.set_facecolor(SURFACE)
    r.setup_axis(ax=ax, facecolor=SURFACE)
    r.draw_circles(ax=ax, facecolor="#f0efec", edgecolor=LINE)
    if values_b is None:
        r.draw_radar(values_a, ax=ax, kwargs_radar={"facecolor": BLUE, "alpha": 0.55},
                     kwargs_rings={"facecolor": SURFACE, "alpha": 0})  # fmt: skip
    else:
        r.draw_radar_compare(values_a, values_b, ax=ax,
                             kwargs_radar={"facecolor": BLUE, "alpha": 0.5},
                             kwargs_compare={"facecolor": ORANGE, "alpha": 0.5})  # fmt: skip
    r.draw_range_labels(ax=ax, fontsize=8, color=INK_MUTED)
    r.draw_param_labels(ax=ax, fontsize=10, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.6)]
    labels = [label_a]
    if values_b is not None:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=ORANGE, alpha=0.6))
        labels.append(label_b)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, labelcolor=INK)
    _title(
        fig, title, subtitle or "Valores por 90 minutos · ejes del percentil 5 al 95 de su posición"
    )
    return fig


def percentile_bounds(df: pd.DataFrame, cols: list[str]) -> tuple[list[float], list[float]]:
    """5th-95th percentile per metric: robust axis limits that outliers can't stretch."""
    low = [float(np.nanpercentile(df[c], 5)) for c in cols]
    high = [float(np.nanpercentile(df[c], 95)) for c in cols]
    high = [h if h > lo else lo + 1e-6 for lo, h in zip(low, high, strict=True)]
    return low, high
