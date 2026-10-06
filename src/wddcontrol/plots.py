"""Figure helpers for the paper: theme, method colors, maps and series."""

from __future__ import annotations

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

THEME = {
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.color": "#d9d8d4",
    "legend.framealpha": 1,
    "legend.facecolor": "white",
    "legend.shadow": True,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "axes.titlelocation": "left",
    "figure.dpi": 150,
    "lines.linewidth": 1.8,
}

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#8a8985"
FAINT = "#e4e3df"
# Categorical slots in fixed order: blue, orange, aqua (the three that stay
# distinguishable for color-vision deficiency when all appear together).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
METHOD_COLOR = {"treated": INK, "contiguous": BLUE, "microsynth": ORANGE, "scan": AQUA, "city": MUTED}
METHOD_LABEL = {"treated": "Treated area", "contiguous": "Contiguous control", "microsynth": "Microsynth",
                "scan": "Scan", "best fit": "Best fit", "ring": "Ring", "city": "City"}
# Sequential single-hue ramps (light -> dark)
GRAYS = LinearSegmentedColormap.from_list("grays", ["#fcfcfb", "#b9b8b3", "#52514e"])
ORANGES = LinearSegmentedColormap.from_list("oranges", ["#fde6dc", "#eb6834", "#7a2d0e"])


def use_theme():
    matplotlib.rcParams.update(THEME)


def _map_axes(ax, title=None):
    ax.set_axis_off()
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)


# ---------------------------------------------------------------------------
# Synthetic grid maps


def grid_panel(ax, nr, nc, base, treated, excluded, sel=None, weights=None, title=None, color=BLUE):
    """One synthetic-world map: log expected crime as a gray base, control cells on top.

    ``sel`` is a boolean cell mask (drawn as filled cells); ``weights`` are
    per-cell weights (drawn on an orange ramp instead of the gray base).
    """
    if weights is not None:
        img = np.ma.masked_less_equal(weights.reshape(nr, nc), 1e-3)
        ax.imshow(np.log(base.reshape(nr, nc)), cmap=GRAYS, alpha=0.35, origin="upper")
        im = ax.imshow(img, cmap=ORANGES, origin="upper", vmin=0, vmax=max(float(np.nanmax(weights)), 1e-3))
    else:
        im = ax.imshow(np.log(base.reshape(nr, nc)), cmap=GRAYS, origin="upper")
        if sel is not None:
            rgba = np.zeros((nr, nc, 4))
            rgba[sel.reshape(nr, nc)] = matplotlib.colors.to_rgba(color, 0.9)
            ax.imshow(rgba, origin="upper")
    for mask, ls, lw in [(excluded, "--", 1.0), (treated, "-", 1.6)]:
        rr, cc = np.nonzero(mask.reshape(nr, nc))
        ax.add_patch(Rectangle((cc.min() - 0.5, rr.min() - 0.5), cc.max() - cc.min() + 1, rr.max() - rr.min() + 1,
                               fill=False, ec=INK, ls=ls, lw=lw))
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    _map_axes(ax, title)
    return im


# ---------------------------------------------------------------------------
# Case study maps


def case_map(ax, units, areas, selections: dict, extent=None, excluded_geoms=None, title=None, pad=1500.0,
             weights=None):
    """Census blocks around the treated area, with control areas filled.

    ``selections`` maps a label to (unit index array, color). ``areas`` is a
    GeoDataFrame with ``part`` = treated / buffer. ``weights`` (unit index,
    weight) shades microsynth weights instead.
    """
    treated = areas.loc[areas["part"] == "treated"]
    buffer = areas.loc[areas["part"] == "buffer"]
    if extent is None:
        geoms = [buffer.geometry.union_all()] + [units.geometry.iloc[idx].union_all() for idx, _ in selections.values()]
        b = np.array([g.bounds for g in geoms])
        extent = (b[:, 0].min() - pad, b[:, 1].min() - pad, b[:, 2].max() + pad, b[:, 3].max() + pad)
    x0, y0, x1, y1 = extent
    near = units.cx[x0:x1, y0:y1]
    near.plot(ax=ax, facecolor="#f4f3f0", edgecolor="#d4d3cf", linewidth=0.15)
    if excluded_geoms is not None and len(excluded_geoms):
        excluded_geoms.plot(ax=ax, facecolor="none", edgecolor=MUTED, hatch="////", linewidth=0.4)
    if weights is not None:
        idx, w = weights
        g = units.iloc[idx].assign(w=w)
        g = g.cx[x0:x1, y0:y1]
        g.plot(ax=ax, column="w", cmap=ORANGES, linewidth=0, vmin=0, vmax=w.max())
    for label, (idx, color) in selections.items():
        units.iloc[idx].plot(ax=ax, facecolor=color, edgecolor="white", linewidth=0.2, alpha=0.9)
    buffer.boundary.plot(ax=ax, color=INK, linestyle="--", linewidth=0.8)
    treated.plot(ax=ax, facecolor=INK, edgecolor=INK, alpha=0.85)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    _map_axes(ax, title)
    return extent


def scale_bar(ax, length_ft=5280, label="1 mile", loc=(0.05, 0.05)):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + loc[0] * (x1 - x0)
    y = y0 + loc[1] * (y1 - y0)
    ax.plot([x, x + length_ft], [y, y], color=INK, lw=2, solid_capstyle="butt")
    ax.text(x + length_ft / 2, y + 0.015 * (y1 - y0), label, ha="center", va="bottom", fontsize=8, color=INK)


# ---------------------------------------------------------------------------
# Series


def smooth_split(y: np.ndarray, n_pre: int, window: int) -> np.ndarray:
    """Centered rolling mean computed separately before and after the intervention."""
    s = pd.Series(np.asarray(y, dtype=float))
    pre = s.iloc[:n_pre].rolling(window, center=True, min_periods=1).mean()
    post = s.iloc[n_pre:].rolling(window, center=True, min_periods=1).mean()
    return pd.concat([pre, post]).to_numpy()


def monthly_panel(ax, dates, series: dict, n_pre: int, window: int = 6, title=None, ylabel="Crimes per month"):
    """Monthly counts (thin) and a centered rolling mean (thick) for each series."""
    dates = pd.to_datetime(dates)
    for name, y in series.items():
        col = METHOD_COLOR.get(name, MUTED)
        top = 4 if name == "treated" else 2  # the treated area's lines stay on top
        ax.plot(dates, y, color=col, lw=0.7, alpha=0.35, zorder=top - 1)
        ax.plot(dates, smooth_split(y, n_pre, window), color=col, lw=2.0 if name == "treated" else 1.8,
                label=METHOD_LABEL.get(name, name), zorder=top)
    ax.axvline(dates[n_pre], color=SECONDARY, lw=1, ls=":")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def cumulative_panel(ax, months, lines: dict, title=None):
    """Running WDD estimates with 95% bands; ``lines`` maps a label to (dict from cumulative_wdd, color)."""
    for label, (r, color) in lines.items():
        ax.fill_between(months, r["low"], r["high"], color=color, alpha=0.15, lw=0)
        ax.plot(months, r["est"], color=color, lw=2, label=label)
    ax.axhline(0, color=SECONDARY, lw=1)
    ax.set_xlabel("Months since the intervention")
    ax.set_ylabel("Cumulative crimes prevented (-)\nor added (+)")
    if title:
        ax.set_title(title)


def legend_handles(items):
    """Patch/line handles from (label, color, kind) tuples."""
    out = []
    for label, color, kind in items:
        if kind == "patch":
            out.append(Patch(facecolor=color, edgecolor="none", label=label))
        elif kind == "outline":
            out.append(Patch(facecolor="none", edgecolor=color, linestyle="--", label=label))
        elif kind == "hatch":
            out.append(Patch(facecolor="none", edgecolor=color, hatch="////", label=label))
        else:
            out.append(Line2D([0], [0], color=color, lw=2, label=label))
    return out
