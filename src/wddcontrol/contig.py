"""Select a contiguous control area whose pre-period crime tracks the treated area.

The control area is a set S of candidate units (census blocks, grid cells)
that is connected in the unit adjacency graph. Write D_t(S) for the gap
between the control's counts and k times the treated counts in pre-period t,

    D_t(S) = sum_{i in S} x_{it} - k * y_t.

Four summaries of the gap are compared with what noise alone would produce
if the control's expected counts were exactly k times the treated area's
(``Criteria``). Noise is Poisson scaled by a dispersion factor phi estimated
from the treated series (quasi-Poisson; phi >= 1):

- fit   F = sum_t |D_t|                  against the noise floor tau,
- block fit, the same on sums of D_t over consecutive blocks of periods
  (6 months by default), against its own noise floor, so the two series
  also track each other over the medium term,
- level L = sum_t D_t                    against z standard deviations,
- slope S = sum_t (t - tbar) D_t         against z standard deviations.

An area passes when all are within their thresholds: its pre-period series
is statistically indistinguishable from the treated area's, month to month,
over the medium term, and in volume and trend. Objectives:

- ``"near"`` (default): among passing areas, the one whose scan window is
  centered nearest the treated area; within a window, the most compact
  (smallest sum of network distances from the window's center).
- ``"fit"``: minimize F with no thresholds. With thousands of candidate
  areas this chases noise in the treated series, so it is kept as a
  comparison.

Search is a network scan plus an integer program within each scan window:

1. Window. From every candidate center c, units are ordered by shortest-path
   (network) distance from c through the adjacency graph, and the window is
   the nearest units holding ``gamma`` times the target crime.
2. Screening. Two cheap fits per window: the best prefix of the distance
   ordering (a network "circle", the pure scan) and a greedy growth from c.
3. Integer program. For the ``top`` windows with the best screening result,
   an integer program picks any subset of the window that includes c and in
   which every selected unit has a selected neighbor closer to c. That
   ordering constraint guarantees a connected area (each unit links back to
   c through ever-closer units), keeps the area compact around c, and needs
   only one constraint per unit, so each program is small.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import highspy
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph


@dataclass
class Criteria:
    """Noise-based thresholds for the gap between a control and k times the treated series."""

    target: np.ndarray  # k * y
    tau: float  # threshold for F = sum |D_t|
    level: float | None  # threshold for |sum D_t| (None: not checked)
    slope: float | None  # threshold for |sum c_t D_t| (None: not checked)
    c: np.ndarray  # centered period index
    blocks: list = field(default_factory=list)  # (B, tau_B): 0/1 block-sum matrix and its noise floor
    phi: float = 1.0  # dispersion factor applied to the Poisson variances

    @classmethod
    def from_series(cls, y: np.ndarray, k: float = 1.0, tau_mult: float = 1.0, z: float | None = 1.0,
                    scales: tuple = (6,), dispersion: float | None = None) -> "Criteria":
        """Thresholds for matching k * y.

        ``scales`` are block lengths (in periods) for the block fits; blocks
        end at the last pre-period, and a leading partial block is dropped.
        ``dispersion`` is phi; None estimates it from ``y`` (``dispersion``).
        """
        y = np.asarray(y, dtype=float)
        P = len(y)
        phi = estimate_dispersion(y) if dispersion is None else float(dispersion)
        v = phi * (k + k**2) * np.maximum(y, 1.0)  # variance of D_t under a perfect match
        c = np.arange(P) - (P - 1) / 2
        tau = tau_mult * float(np.sqrt(2 * v / np.pi).sum())
        level = None if z is None else z * float(np.sqrt(v.sum()))
        slope = None if z is None else z * float(np.sqrt((c**2 * v).sum()))
        blocks = []
        for b in scales:
            nb = P // b
            if nb < 2:
                continue
            B = np.zeros((nb, P))
            for j in range(nb):
                B[j, P - (nb - j) * b:P - (nb - j - 1) * b] = 1.0
            blocks.append((B, tau_mult * float(np.sqrt(2 * (B @ v) / np.pi).sum())))
        return cls(target=k * y, tau=tau, level=level, slope=slope, c=c, blocks=blocks, phi=phi)

    def violation(self, series: np.ndarray) -> np.ndarray:
        """Largest ratio of a gap summary to its threshold (<= 1 passes). Works on (P,) or (n, P)."""
        D = np.asarray(series, dtype=float) - self.target
        v = np.abs(D).sum(axis=-1) / self.tau
        for B, tb in self.blocks:
            v = np.maximum(v, np.abs(D @ B.T).sum(axis=-1) / tb)
        if self.level is not None:
            v = np.maximum(v, np.abs(D.sum(axis=-1)) / self.level)
            v = np.maximum(v, np.abs(D @ self.c) / self.slope)
        return v

    def fit(self, series: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(series, dtype=float) - self.target).sum(axis=-1)


def estimate_dispersion(y: np.ndarray, window: int = 7) -> float:
    """Pearson dispersion of a count series around its centered moving average (at least 1).

    phi = sum (y_t - m_t)^2 / m_t / (P (1 - 1/window)), where m_t is the
    ``window``-period centered moving average; the denominator discounts
    the degrees of freedom the moving average uses.
    """
    y = np.asarray(y, dtype=float)
    P = len(y)
    if P < 2 * window:
        return 1.0
    kern = np.ones(window)
    m = np.convolve(y, kern, mode="same") / np.convolve(np.ones(P), kern, mode="same")
    ok = m > 0
    phi = ((y[ok] - m[ok]) ** 2 / m[ok]).sum() / (ok.sum() * (1 - 1 / window))
    return float(max(phi, 1.0))


def noise_floor(y: np.ndarray, k: float = 1.0) -> float:
    """Expected fit F for a control with exactly k times the treated area's expected counts.

    With Poisson counts, D_t has variance (k + k^2) mu_t, so its expected
    absolute value is sqrt(2 (k + k^2) mu_t / pi); mu_t is estimated by y_t.
    """
    return Criteria.from_series(y, k, dispersion=1.0).tau


@dataclass
class Window:
    center: int  # node index of the center (also nodes[0])
    nodes: np.ndarray  # node indices in order of network distance from the center
    dist: np.ndarray  # network distance of each node from the center
    closer: sparse.csr_matrix  # closer[p, q] = 1 if window node q neighbors p and is closer to the center
    enough: bool  # True when the window holds at least gamma times the target crime


@dataclass
class Selection:
    nodes: np.ndarray  # selected node indices
    cost: float  # fit F
    viol: float  # largest gap summary relative to its threshold (<= 1 passes)
    area: float  # summed area of the selected units
    center: int
    method: str
    status: str = ""
    gap: float = 0.0
    seconds: float = 0.0


@dataclass
class SearchResult:
    best: Selection
    scan: Selection  # best pure-scan (prefix) area
    greedy: Selection  # best greedy area
    criteria: Criteria
    ilp: list[Selection] = field(default_factory=list)  # integer program solutions, best first
    seconds: dict = field(default_factory=dict)
    n_windows: int = 0

    @property
    def tau(self) -> float:
        return self.criteria.tau


def distance_graph(adj: sparse.csr_matrix, xy: np.ndarray) -> sparse.csr_matrix:
    """Adjacency weighted by the straight-line distance between unit centers."""
    coo = sparse.triu(adj, k=1).tocoo()
    d = np.hypot(*(xy[coo.row] - xy[coo.col]).T)
    d = np.maximum(d, 1e-6)
    g = sparse.coo_matrix((d, (coo.row, coo.col)), shape=adj.shape)
    return (g + g.T).tocsr()


# ---------------------------------------------------------------------------
# Windows


def _window(adj: sparse.csr_matrix, center: int, row: np.ndarray, tot: np.ndarray, need: float, max_units: int) -> Window:
    reached = np.flatnonzero(np.isfinite(row))
    order = reached[np.argsort(row[reached], kind="stable")]
    cum = np.cumsum(tot[order])
    m = int(np.searchsorted(cum, need)) + 1  # first prefix holding `need`
    enough = m <= len(order)
    m = min(m, len(order), max_units)
    nodes = order[:m]
    dist = row[nodes]
    sub = adj[nodes][:, nodes].tocoo()
    keep = dist[sub.col] < dist[sub.row]
    closer = sparse.csr_matrix(
        (np.ones(keep.sum(), dtype=np.int8), (sub.row[keep], sub.col[keep])), shape=(m, m)
    )
    return Window(center=center, nodes=nodes, dist=dist, closer=closer, enough=enough)


def iter_windows(graph: sparse.csr_matrix, centers: np.ndarray, tot: np.ndarray, need: float,
                 radius: float, max_units: int = 600, batch: int = 256):
    """Yield one Window per center.

    Dijkstra runs in batches out to ``radius``; centers whose window does not
    reach ``need`` crimes are retried with a doubled radius (up to 4 times).
    """
    adj = (graph > 0).astype(np.int8).tocsr()
    todo = np.asarray(centers)
    r = radius
    for attempt in range(5):
        retry = []
        for s in range(0, len(todo), batch):
            idx = todo[s:s + batch]
            dmat = csgraph.dijkstra(graph, directed=False, indices=idx, limit=r)
            for c, row in zip(idx, dmat):
                w = _window(adj, int(c), row, tot, need, max_units)
                if w.enough or attempt == 4 or len(w.nodes) >= max_units:
                    yield w
                else:
                    retry.append(int(c))
        if not retry:
            return
        todo = np.array(retry)
        r *= 2


# ---------------------------------------------------------------------------
# Screening fits


def scan_prefix(Xw: np.ndarray, crit: Criteria, thresholds: bool) -> int:
    """Size of the best prefix (network circle) of a distance-ordered window.

    With ``thresholds`` the smallest passing prefix (the smallest area, since
    prefix areas only grow), else the prefix with the smallest violation;
    without, the prefix with the best fit.
    """
    cum = np.cumsum(Xw, axis=0)
    if thresholds:
        viol = crit.violation(cum)
        ok = np.flatnonzero(viol <= 1)
        return int(ok[0]) + 1 if len(ok) else int(np.argmin(viol)) + 1
    return int(np.argmin(crit.fit(cum))) + 1


def greedy_grow(Xw: np.ndarray, closer: sparse.csr_matrix, crit: Criteria, thresholds: bool) -> np.ndarray:
    """Grow from the center, adding the eligible unit that most improves the score.

    A unit is eligible once one of its closer neighbors is selected, so the
    result satisfies the same ordering constraint as the integer program.
    The score is the violation with ``thresholds`` (growth stops once the
    area passes), else the fit.
    """
    score = crit.violation if thresholds else crit.fit
    m = len(Xw)
    sel = np.zeros(m, dtype=bool)
    sel[0] = True
    cur = Xw[0].astype(float).copy()
    best = float(score(cur))
    dep = closer.T.tocsr()  # dep[q] = window units that have q as a closer neighbor
    n_sel_closer = np.zeros(m, dtype=np.int32)
    n_sel_closer[dep.indices[dep.indptr[0]:dep.indptr[1]]] += 1
    while not (thresholds and best <= 1):
        elig = np.flatnonzero(~sel & (n_sel_closer > 0))
        if len(elig) == 0:
            break
        new = score(cur[None, :] + Xw[elig])
        b = int(np.argmin(new))
        if new[b] >= best - 1e-9:
            break
        q = elig[b]
        sel[q] = True
        cur += Xw[q]
        best = float(new[b])
        n_sel_closer[dep.indices[dep.indptr[q]:dep.indptr[q + 1]]] += 1
    return np.flatnonzero(sel)


def thin_centers(xy: np.ndarray, weight: np.ndarray, spacing: float) -> np.ndarray:
    """One center per ``spacing`` x ``spacing`` grid cell: the unit with the largest weight."""
    cell = np.floor(xy / spacing).astype(np.int64)
    key = (cell[:, 0] - cell[:, 0].min()) * (cell[:, 1].max() - cell[:, 1].min() + 1) + (cell[:, 1] - cell[:, 1].min())
    order = np.lexsort((-weight, key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    return np.sort(order[first])


# ---------------------------------------------------------------------------
# Integer program


def solve_window(Xw: np.ndarray, dw: np.ndarray, closer: sparse.csr_matrix, crit: Criteria, thresholds: bool,
                 start: np.ndarray | None = None, eps: float = 1e-3, time_limit: float = 2.0,
                 degree: np.ndarray | None = None) -> tuple[np.ndarray, str, float]:
    """Best subset of the window under the closer-neighbor ordering constraint.

    Columns: z_0..z_{m-1} (binary, z_0 = 1 is the center), over_t and under_t
    (continuous, >= 0, so over_t - under_t = D_t), with ``thresholds`` an
    absolute block gap e_j >= |sum_{t in block j} D_t| per block, and an
    elastic slack s >= 0.

    - Without thresholds: minimize sum(over + under) + eps * sum(d_i z_i).
    - With thresholds: minimize perimeter + sum(d_i z_i) + big * s + eps * F / tau,
      where d_i is unit i's network distance from the center (rescaled to
      mean 1), the p-median measure of compactness, and the perimeter is
      the number of adjacencies between a selected and an unselected unit,
      sum_i deg_i z_i - 2 sum_{ij} w_ij with w_ij <= z_i, w_ij <= z_j for
      each adjacent pair in the window (``degree`` is each unit's number of
      candidate neighbors; perimeter is skipped when it is None), subject to
      F / tau - s <= 1, sum_j e_j / tau_B - s <= 1, |L| / level - s <= 1
      and |S| / slope - s <= 1. The slack keeps the program feasible when no
      subset passes; ``big`` is 100 times the window's total distance, so a
      subset that passes beats any that misses by more than 1%.

    Returns (positions, solver status, MIP gap).
    """
    m, P = Xw.shape
    dpos = dw[dw > 0]
    dist = dw / (dpos.mean() if len(dpos) else 1.0)
    target = crit.target
    nblk = [B.shape[0] for B, _ in crit.blocks] if thresholds else []
    perim = thresholds and degree is not None
    if perim:
        ew = sparse.triu((closer + closer.T) > 0, k=1).tocoo()
        n_e = ew.nnz
    else:
        n_e = 0
    ncol = m + 2 * P + sum(nblk) + n_e + int(thresholds)
    o, u = m + np.arange(P), m + P + np.arange(P)
    # Fit rows: sum_j x_jt z_j - over_t + under_t = target_t
    r_fit = np.repeat(np.arange(P), m)
    c_fit = np.tile(np.arange(m), P)
    v_fit = Xw.T.ravel().astype(float)
    nz = v_fit != 0
    rows = [r_fit[nz], np.arange(P), np.arange(P)]
    cols = [c_fit[nz], o, u]
    vals = [v_fit[nz], -np.ones(P), np.ones(P)]
    # Ordering rows: z_p - sum_{q closer neighbor of p} z_q <= 0 for p >= 1
    cl = closer.tocoo()
    rows += [P + np.arange(m - 1), P + cl.row - 1]
    cols += [np.arange(1, m), cl.col]
    vals += [np.ones(m - 1), -np.ones(cl.nnz)]
    nrow = P + m - 1
    row_lo = [target.astype(float), np.full(m - 1, -np.inf)]
    row_hi = [target.astype(float), np.zeros(m - 1)]
    if thresholds:
        s = ncol - 1

        def add_row(idx, coef):
            nonlocal nrow
            rows.append(np.full(len(idx) + 1, nrow))
            cols.append(np.append(idx, s))
            vals.append(np.append(coef, -1.0))
            row_lo.append([-np.inf])
            row_hi.append([1.0])
            nrow += 1

        add_row(np.concatenate([o, u]), np.ones(2 * P) / crit.tau)
        e0 = m + 2 * P
        for (B, tb), nb in zip(crit.blocks, nblk):
            e = e0 + np.arange(nb)
            for j in range(nb):
                # e_j >= +-sum_{t in block j} (over_t - under_t)
                ts = np.flatnonzero(B[j])
                for sign in (1.0, -1.0):
                    rows.append(np.full(2 * len(ts) + 1, nrow))
                    cols.append(np.concatenate([o[ts], u[ts], [e[j]]]))
                    vals.append(np.concatenate([np.full(len(ts), sign), np.full(len(ts), -sign), [-1.0]]))
                    row_lo.append([-np.inf])
                    row_hi.append([0.0])
                    nrow += 1
            add_row(e, np.ones(nb) / tb)
            e0 += nb
        if crit.level is not None:
            for sign in (1.0, -1.0):
                add_row(np.concatenate([o, u]), sign * np.concatenate([np.ones(P), -np.ones(P)]) / crit.level)
                add_row(np.concatenate([o, u]), sign * np.concatenate([crit.c, -crit.c]) / crit.slope)
        zcost = dist.copy()
        wcost = np.zeros(n_e)
        if perim:
            # internal edge w_e <= z_i and w_e <= z_j; perimeter = sum deg_i z_i - 2 sum w_e
            wc = e0 + np.arange(n_e)
            for ends in (ew.row, ew.col):
                rows.append(np.repeat(nrow + np.arange(n_e), 2))
                cols.append(np.column_stack([wc, ends]).ravel())
                vals.append(np.tile([1.0, -1.0], n_e))
                row_lo.append(np.full(n_e, -np.inf))
                row_hi.append(np.zeros(n_e))
                nrow += n_e
            zcost = zcost + degree.astype(float)
            wcost = np.full(n_e, -2.0)
        big = 100.0 * max(zcost.sum(), 1.0)
        cost = np.concatenate([zcost, np.full(2 * P, eps / crit.tau), np.zeros(sum(nblk)), wcost, [big]])
    else:
        cost = np.concatenate([eps * dist, np.ones(2 * P)])
    A = sparse.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nrow, ncol)
    )

    lp = highspy.HighsLp()
    lp.num_col_ = ncol
    lp.num_row_ = nrow
    lp.col_cost_ = cost
    lo = np.zeros(ncol)
    lo[0] = 1.0
    hi = np.concatenate([np.ones(m), np.full(ncol - m, np.inf)])
    lp.col_lower_ = lo
    if n_e:
        hi[m + 2 * P + sum(nblk):m + 2 * P + sum(nblk) + n_e] = 1.0
    lp.col_upper_ = hi
    lp.row_lower_ = np.concatenate(row_lo)
    lp.row_upper_ = np.concatenate(row_hi)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    lp.integrality_ = [highspy.HighsVarType.kInteger] * m + [highspy.HighsVarType.kContinuous] * (ncol - m)

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 1e-3)
    h.passModel(lp)
    if start is not None:
        z = np.zeros(m)
        z[start] = 1.0
        D = z @ Xw - target
        x0 = np.concatenate([z, np.maximum(D, 0), np.maximum(-D, 0)])
        if thresholds:
            for B, _ in crit.blocks:
                x0 = np.concatenate([x0, np.abs(B @ D)])
            if n_e:
                x0 = np.concatenate([x0, np.minimum(z[ew.row], z[ew.col])])
            x0 = np.append(x0, max(float(crit.violation(z @ Xw)) - 1.0, 0.0))
        h.setSolution(ncol, np.arange(ncol, dtype=np.int32), x0)
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    x = np.asarray(h.getSolution().col_value)
    if len(x) != ncol:  # no solution found (cannot happen with a warm start)
        return (np.asarray(start) if start is not None else np.array([0])), status, np.inf
    return np.flatnonzero(x[:m] > 0.5), status, float(h.getInfo().mip_gap)


# ---------------------------------------------------------------------------
# Full search


def _rank_key(cost: float, viol: float, area: float, objective: str, dist: float = 0.0) -> tuple:
    """Sort key: passing areas by window distance (then size), then the rest by violation."""
    if objective == "fit":
        return (0, cost, area)
    if viol <= 1 + 1e-9:
        return (0, dist, area, cost)
    return (1, viol, dist, area)


def search(X: np.ndarray, y: np.ndarray, graph: sparse.csr_matrix, candidates: np.ndarray, k: float = 1.0,
           objective: str = "near", area: np.ndarray | None = None, tau_mult: float = 1.0, z: float | None = 1.0,
           scales: tuple = (6,), dispersion: float | None = None, perimeter: bool = True,
           gamma: float = 3.0, radius: float | None = None, centers: np.ndarray | None = None, top: int = 50,
           time_limit: float = 2.0, refine: int = 8, refine_time: float = 10.0, max_units: int = 600,
           eps: float = 1e-3, dist_to_treated: np.ndarray | None = None, xy: np.ndarray | None = None,
           center_spacing: float | None = None, verbose: bool = False) -> SearchResult:
    """Search for a contiguous control area that tracks ``k * y``.

    Parameters
    ----------
    X : (n_units, P) pre-period counts for every unit.
    y : (P,) treated area pre-period counts.
    graph : distance-weighted adjacency over all units (``distance_graph``).
    candidates : boolean mask of units allowed in the control area (outside
        the treated area and its buffer). Other units are removed from the
        graph, so the control area cannot pass through them.
    k : the control area tracks k times the treated series.
    objective : ``"near"`` or ``"fit"`` (see the module docstring).
    area : land area of each unit (default 1 per unit), reported and used to break ties.
    tau_mult : the fit threshold is ``tau_mult`` times the Poisson noise floor.
    z : level and slope thresholds in Poisson standard deviations (None: not checked).
    scales : block lengths (in periods) for the block fits.
    dispersion : phi for the thresholds (None: estimated from ``y``; 1: Poisson).
    perimeter : refine the best ``refine`` passing areas, adding the area's
        perimeter (in adjacencies) to the compactness objective, warm-started
        from the first-stage solution with ``refine_time`` seconds each.
    gamma : windows hold gamma times the target crime.
    radius : starting Dijkstra radius (graph units); defaults to a radius
        expected to hold ``gamma`` times the target at average density.
    centers : node indices to scan from (default: every candidate unit with crime,
        thinned to one per ``center_spacing`` grid cell of ``xy`` coordinates when given).
    top : number of windows passed to the integer program.
    dist_to_treated : distance of each unit from the treated area (``"near"`` only).
    """
    t0 = time.perf_counter()
    crit = Criteria.from_series(y, k, tau_mult, z, scales, dispersion)
    thresholds = objective != "fit"
    if objective not in ("near", "fit"):
        raise ValueError(f"unknown objective {objective!r}")
    if objective == "near" and dist_to_treated is None:
        raise ValueError("objective 'near' needs dist_to_treated")
    need = gamma * crit.target.sum()
    cand = np.flatnonzero(candidates)
    sub = graph[cand][:, cand].tocsr()
    Xc = np.asarray(X[cand], dtype=float)
    ac = np.ones(len(cand)) if area is None else np.asarray(area, dtype=float)[cand]
    dc_all = np.zeros(len(cand)) if dist_to_treated is None else np.asarray(dist_to_treated, dtype=float)[cand]
    tot = Xc.sum(axis=1)
    deg = np.diff(sub.indptr)
    if centers is None:
        centers_local = np.flatnonzero(tot > 0)
        if center_spacing is not None:
            xy_c = np.asarray(xy, dtype=float)[cand][centers_local]
            centers_local = centers_local[thin_centers(xy_c, tot[centers_local], center_spacing)]
    else:
        lookup = np.full(len(X), -1)
        lookup[cand] = np.arange(len(cand))
        centers_local = lookup[np.asarray(centers)]
        centers_local = centers_local[centers_local >= 0]
    if radius is None:
        # mean edge length times the hop radius expected to hold `need` crimes
        mean_edge = sub.data.mean() if sub.nnz else 1.0
        units_needed = need / max(tot.mean(), 1e-9)
        radius = mean_edge * np.sqrt(units_needed / np.pi) * 1.5

    def make(nodes_local, center_local, method, **kw):
        series = Xc[nodes_local].sum(axis=0)
        return Selection(nodes=cand[nodes_local], cost=float(crit.fit(series)), viol=float(crit.violation(series)),
                         area=float(ac[nodes_local].sum()), center=int(cand[center_local]), method=method, **kw)

    def key(sel_, center_local):
        return _rank_key(sel_.cost, sel_.viol, sel_.area, objective, float(dc_all[center_local]))

    scan_best = greedy_best = None
    kept = []  # (key, window, start positions)
    n_windows = 0
    for w in iter_windows(sub, centers_local, tot, need, radius, max_units):
        n_windows += 1
        Xw = Xc[w.nodes]
        m = scan_prefix(Xw, crit, thresholds)
        g_pos = greedy_grow(Xw, w.closer, crit, thresholds)
        s_sel = make(w.nodes[:m], w.center, "scan")
        g_sel = make(w.nodes[g_pos], w.center, "greedy")
        s_key, g_key = key(s_sel, w.center), key(g_sel, w.center)
        if scan_best is None or s_key < scan_best[0]:
            scan_best = (s_key, s_sel)
        if greedy_best is None or g_key < greedy_best[0]:
            greedy_best = (g_key, g_sel)
        kept.append((s_key, w, np.arange(m)) if s_key < g_key else (g_key, w, g_pos))
        if len(kept) > 4 * top:
            kept.sort(key=lambda v: v[0])
            kept = kept[:top]
    t1 = time.perf_counter()

    kept.sort(key=lambda v: v[0])
    sols = []
    for _, w, start in kept[:top]:
        ts = time.perf_counter()
        pos, status, gap = solve_window(Xc[w.nodes], w.dist, w.closer, crit, thresholds, start=start,
                                        eps=eps, time_limit=time_limit)
        sols.append((make(w.nodes[pos], w.center, "ilp", status=status, gap=gap, seconds=time.perf_counter() - ts),
                     w, pos))
    sols.sort(key=lambda v: key(v[0], v[1].center))
    # Second stage: smooth the shape of the best passing areas (fill holes,
    # trim tendrils) by adding the perimeter to the objective.
    if perimeter and thresholds:
        for i, (sel_, w, pos) in enumerate(sols[:refine]):
            if sel_.viol > 1 + 1e-9:
                continue
            ts = time.perf_counter()
            pos2, status, gap = solve_window(Xc[w.nodes], w.dist, w.closer, crit, True, start=pos, eps=eps,
                                             time_limit=refine_time, degree=deg[w.nodes])
            new = make(w.nodes[pos2], w.center, "ilp", status=status, gap=gap,
                       seconds=sel_.seconds + time.perf_counter() - ts)
            if new.viol <= 1 + 1e-9:
                sols[i] = (new, w, pos2)
        sols.sort(key=lambda v: key(v[0], v[1].center))
    sols = [v[0] for v in sols]
    t2 = time.perf_counter()

    best = sols[0] if sols else greedy_best[1]
    if verbose:
        print(f"{n_windows} windows screened in {t1 - t0:.1f}s; {len(sols)} integer programs in {t2 - t1:.1f}s; "
              f"tau {crit.tau:.0f}; scan fit {scan_best[1].cost:.0f} viol {scan_best[1].viol:.2f} "
              f"area {scan_best[1].area:.0f}; ilp fit {best.cost:.0f} viol {best.viol:.2f} area {best.area:.0f}")
    return SearchResult(best=best, scan=scan_best[1], greedy=greedy_best[1], criteria=crit, ilp=sols,
                        n_windows=n_windows, seconds={"screen": t1 - t0, "ilp": t2 - t1, "total": t2 - t0})
