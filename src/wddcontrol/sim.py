"""Synthetic grid worlds with spatially varying crime levels and trends.

Each grid cell's monthly expected count is

    log lambda_it = mu + a_i + b_i * (t / 12) + c_i * bump(t) + s(t),

where a_i (level), b_i (yearly trend) and c_i (a temporary rise and fall that
peaks mid pre-period) are smooth Gaussian random fields, so neighborhoods
share levels and trends but trends drift across the map, and s(t) is a common
seasonal cycle. Counts are Poisson with gamma-distributed overdispersion
(negative binomial).

Two scenarios differ in how local trends are:

- ``"smooth"``: the trend field varies only over long distances, and the
  treated hot spot is chosen on its crime level alone, so its neighbors
  share its trend.
- ``"local"``: the trend adds a short-range field (a few cells across), so
  adjacent places can follow different paths, and the treated hot spot is
  chosen among those where crime was rising, as interventions often are.

With ``coarse = f`` the world is simulated on the fine grid and then each f x f
block of cells is summed into one coarse cell (the treated block and buffer
are aligned to the coarse cells), giving fewer, larger units of analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter


@dataclass
class World:
    nr: int
    nc: int
    lam: np.ndarray  # (n_cells, n_months) expected counts without treatment
    counts: np.ndarray  # (n_cells, n_months) observed counts (treated cells include the effect)
    treated: np.ndarray  # boolean mask of treated cells
    excluded: np.ndarray  # boolean mask of treated + buffer cells
    n_pre: int
    n_post: int
    effect: float  # proportional change in treated cells after the intervention
    xy: np.ndarray  # (n_cells, 2) cell centers
    adj: sparse.csr_matrix  # rook adjacency


def grid_adjacency(nr: int, nc: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    idx = np.arange(nr * nc).reshape(nr, nc)
    right = (idx[:, :-1].ravel(), idx[:, 1:].ravel())
    down = (idx[:-1, :].ravel(), idx[1:, :].ravel())
    r = np.concatenate([right[0], down[0]])
    c = np.concatenate([right[1], down[1]])
    n = nr * nc
    adj = sparse.coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    adj = ((adj + adj.T) > 0).astype(np.int8).tocsr()
    rr, cc = np.divmod(np.arange(n), nc)
    xy = np.column_stack([cc + 0.5, nr - rr - 0.5]).astype(float)
    return adj, xy


def smooth_field(rng: np.random.Generator, nr: int, nc: int, scale: float) -> np.ndarray:
    """Standardized Gaussian random field with correlation length ~ ``scale`` cells."""
    f = gaussian_filter(rng.standard_normal((nr, nc)), sigma=scale, mode="wrap")
    return ((f - f.mean()) / f.std()).ravel()


def simulate(rng: np.random.Generator, scenario: str = "smooth", nr: int = 40, nc: int = 40, n_pre: int = 60,
             n_post: int = 12, block: int = 4, buffer: int = 2, effect: float = -0.2, mu: float = -1.0,
             sd_level: float = 1.0, sd_trend: float = 0.08, sd_local: float = 0.12, sd_bump: float = 0.25,
             sd_cell: float = 0.3, season: float = 0.15, shape: float = 10.0, hot_quantile: float = 0.9,
             rising_quantile: float = 0.75, coarse: int = 1) -> World:
    """One synthetic world with a ``block`` x ``block`` treated hot spot.

    The treated block is drawn at random among blocks whose expected
    pre-period crime is above ``hot_quantile`` (interventions target hot
    spots) and at least ``buffer + 6`` cells from the edge; in the
    ``"local"`` scenario, also among the top ``1 - rising_quantile`` of
    those by trend. Cells within ``buffer`` cells (Chebyshev distance) of the
    block are excluded from the control pool.
    """
    n = nr * nc
    T = n_pre + n_post
    t = np.arange(T)
    a = sd_level * smooth_field(rng, nr, nc, 4) + sd_cell * rng.standard_normal(n)
    b = sd_trend * smooth_field(rng, nr, nc, 6)
    local = np.zeros(n)
    if scenario == "local":
        local = sd_local * smooth_field(rng, nr, nc, 1.5)
        b = b + local
    c = sd_bump * smooth_field(rng, nr, nc, 5)
    bump = np.exp(-0.5 * ((t - n_pre / 2) / (n_pre / 6)) ** 2)
    s = season * np.sin(2 * np.pi * t / 12)
    lam = np.exp(mu + a[:, None] + b[:, None] * (t[None, :] / 12) + c[:, None] * bump[None, :] + s[None, :])

    # Treated block: a random hot spot away from the edges.
    edge = buffer + 6
    pre_tot = lam[:, :n_pre].sum(axis=1).reshape(nr, nc)
    win = np.lib.stride_tricks.sliding_window_view(pre_tot, (block, block)).sum(axis=(2, 3))
    ok = np.zeros_like(win, dtype=bool)
    ok[edge:nr - block - edge + 1, edge:nc - block - edge + 1] = True
    if coarse > 1:  # align the treated block to the coarse cells
        ok[np.arange(ok.shape[0]) % coarse != 0, :] = False
        ok[:, np.arange(ok.shape[1]) % coarse != 0] = False
    hot = ok & (win >= np.quantile(win[ok], hot_quantile))
    if scenario == "local":
        # rising relative to its surroundings: high short-range trend component
        lwin = np.lib.stride_tricks.sliding_window_view(local.reshape(nr, nc), (block, block)).mean(axis=(2, 3))
        hot &= lwin >= np.quantile(lwin[hot], rising_quantile)
    r0, c0 = map(int, rng.choice(np.argwhere(hot)))
    rr, cc = np.divmod(np.arange(n), nc)
    dr = np.maximum(np.maximum(r0 - rr, rr - (r0 + block - 1)), 0)
    dc = np.maximum(np.maximum(c0 - cc, cc - (c0 + block - 1)), 0)
    cheb = np.maximum(dr, dc)
    treated = cheb == 0
    excluded = cheb <= buffer

    mult = np.ones((n, T))
    mult[np.ix_(treated, np.arange(n_pre, T))] = 1 + effect
    noise = rng.gamma(shape, 1 / shape, size=(n, T))
    counts = rng.poisson(lam * mult * noise)
    if coarse > 1:
        f = coarse
        cr, cc_ = nr // f, nc // f
        key = (rr // f) * cc_ + (cc // f)
        agg = sparse.csr_matrix((np.ones(n), (key, np.arange(n))), shape=(cr * cc_, n))
        lam, counts = agg @ lam, np.asarray(agg @ counts).astype(np.int64)
        treated = np.asarray(agg @ treated.astype(float)).ravel() > 0
        excluded = np.asarray(agg @ excluded.astype(float)).ravel() > 0
        nr, nc = cr, cc_
    adj, xy = grid_adjacency(nr, nc)
    return World(nr=nr, nc=nc, lam=lam, counts=counts, treated=treated, excluded=excluded, n_pre=n_pre,
                 n_post=n_post, effect=effect, xy=xy, adj=adj)


def grid_edge(nr: int, nc: int) -> np.ndarray:
    """Cells on the grid's outer edge."""
    rr, cc = np.divmod(np.arange(nr * nc), nc)
    return (rr == 0) | (rr == nr - 1) | (cc == 0) | (cc == nc - 1)


def aggregate(counts: np.ndarray, months: int) -> np.ndarray:
    """Sum consecutive blocks of ``months`` columns (trailing partial block dropped)."""
    n, T = counts.shape
    P = T // months
    return counts[:, T - P * months:].reshape(n, P, months).sum(axis=2)
