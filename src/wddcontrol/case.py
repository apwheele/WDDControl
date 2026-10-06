"""Case study setup: treated area, buffer, candidate pool, and period counts."""

from __future__ import annotations

import time
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy import sparse

from . import contig, geo, microsynth
from .wdd import wdd, wdd_weighted


@dataclass
class Case:
    name: str
    units: gpd.GeoDataFrame
    graph: sparse.csr_matrix  # distance-weighted adjacency
    treated_geom: object  # shapely geometry of the treated area
    treated_units: np.ndarray  # boolean: units inside the treated area (for microsynth's unit count)
    candidates: np.ndarray  # boolean: units allowed in a control area
    dist: np.ndarray  # distance of each unit from the treated area
    start: pd.Timestamp  # intervention date
    months: int  # period length in months
    n_pre: int  # matching periods before the intervention
    n_post: int  # periods after
    periods: pd.DataFrame  # period index, start, end dates
    X: dict  # crime type -> (n_units, n_pre + n_post) counts
    y: dict  # crime type -> (n_pre + n_post,) treated area counts
    crime: pd.DataFrame  # incidents: date, crime_type, unit, in_treated


def build_case(name: str, units: gpd.GeoDataFrame, adj: sparse.csr_matrix, crime: pd.DataFrame, treated_geom,
               start: str, months: int, n_pre: int, n_post: int, buffer: float, exclude_geoms=(),
               treated_count_buffer: float = 100.0) -> Case:
    """Counts by unit and period, and the candidate pool for the control area.

    Treated counts are incidents inside ``treated_geom`` buffered by
    ``treated_count_buffer`` (to catch incidents on boundary streets).
    Candidate units exclude everything within ``buffer`` of the treated area
    and any unit intersecting ``exclude_geoms`` (e.g. other improvement
    districts receiving similar services).
    """
    start = pd.Timestamp(start)
    graph = contig.distance_graph(adj, units[["x", "y"]].to_numpy())
    dist = units.distance(treated_geom).to_numpy()
    candidates = dist > buffer
    for g in exclude_geoms:
        candidates &= ~units.intersects(g).to_numpy()
    rep = units.representative_point()
    treated_units = shapely.contains(treated_geom, rep.values)

    period = geo.period_index(crime["date"], start, months)
    in_treated = geo.points_in(crime, treated_geom.buffer(treated_count_buffer), units.crs)
    X, y = {}, {}
    for ctype in sorted(crime["crime_type"].unique()):
        m = (crime["crime_type"] == ctype).to_numpy()
        X[ctype] = geo.unit_counts(crime["unit"].to_numpy()[m], period[m], len(units), -n_pre, n_post - 1)
        p = period[m & in_treated]
        p = p[(p >= -n_pre) & (p < n_post)]
        y[ctype] = np.bincount(p + n_pre, minlength=n_pre + n_post)
    idx = np.arange(-n_pre, n_post)
    periods = pd.DataFrame({
        "period": idx,
        "start": [start + pd.DateOffset(months=int(i * months)) for i in idx],
        "end": [start + pd.DateOffset(months=int((i + 1) * months)) - pd.Timedelta(days=1) for i in idx],
    })
    inc = crime[["date", "crime_type", "unit"]].assign(in_treated=in_treated)
    return Case(name=name, units=units, graph=graph, treated_geom=treated_geom, treated_units=treated_units,
                candidates=candidates, dist=dist, start=start, months=months, n_pre=n_pre, n_post=n_post,
                periods=periods, X=X, y=y, crime=inc)


def monthly(case: Case, ctype: str, n_before: int, n_after: int, masks: dict, weights: dict,
            candidates: np.ndarray | None = None) -> pd.DataFrame:
    """Monthly counts for the treated area and each control, ``n_before`` months before to ``n_after`` after.

    ``masks`` maps a name to a boolean unit mask (summed counts); ``weights``
    maps a name to per-unit weights over the candidate units (weighted
    counts, plus a ``{name}_var`` column with the Poisson variance sum w^2 x).
    Months run from the intervention's day of month. ``candidates`` is the
    unit mask the weights refer to (default: the case's candidates).
    """
    inc = case.crime[case.crime["crime_type"] == ctype]
    month = geo.period_index(inc["date"], case.start, 1)
    keep = (month >= -n_before) & (month < n_after)
    inc, month = inc[keep], month[keep]
    idx = np.arange(-n_before, n_after)
    out = pd.DataFrame({"month": idx, "start": [case.start + pd.DateOffset(months=int(i)) for i in idx]})
    out["treated"] = np.bincount(month[inc["in_treated"].to_numpy()] + n_before, minlength=len(idx))
    U = geo.unit_counts(inc["unit"].to_numpy(), month, len(case.units), -n_before, n_after - 1)
    for name, m in masks.items():
        out[name] = U[m].sum(axis=0)
    cand = np.flatnonzero(case.candidates if candidates is None else candidates)
    for name, w in weights.items():
        out[name] = w @ U[cand]
        out[f"{name}_var"] = (w**2) @ U[cand]
    return out


def ring_units(case: Case, inner: float, outer: float, candidates: np.ndarray | None = None) -> np.ndarray:
    """Candidate units between ``inner`` and ``outer`` from the treated area (a donut control)."""
    cand = case.candidates if candidates is None else candidates
    return cand & (case.dist > inner) & (case.dist <= outer)


def evaluate(case: Case, ctype: str, n_wdd: int | None = None, objective: str = "near", top: int = 50,
             time_limit: float = 2.0, ring: tuple[float, float] | None = None, max_share: float | None = 0.2,
             center_spacing: float | None = 800.0, verbose: bool = True) -> dict:
    """Run the contiguous search and the comparison controls for one crime type.

    The search matches the ``n_pre`` pre-period counts. The WDD compares the
    ``n_wdd`` periods before the intervention with the ``n_wdd`` periods after
    (default: all post periods). Returns a dict with the search result,
    per-method WDD estimates, and control series.

    ``max_share`` drops candidate units holding more than that share of the
    treated area's pre-period crime from every method's pool, so no single
    place (one big store, one apartment complex) dominates a control.
    ``center_spacing`` thins scan centers to one per grid cell of that size
    (feet), since windows centered on neighboring blocks nearly coincide.
    """
    n_wdd = case.n_post if n_wdd is None else n_wdd
    X, yall = case.X[ctype], case.y[ctype]
    pre, post = slice(0, case.n_pre), slice(case.n_pre, case.n_pre + case.n_post)
    Xp, yp = X[:, pre], yall[pre]
    w0 = slice(case.n_pre - n_wdd, case.n_pre)
    w1 = slice(case.n_pre, case.n_pre + n_wdd)
    T0, T1 = yall[w0].sum(), yall[w1].sum()
    area = case.units["ALAND20"].to_numpy().astype(float)
    cand = case.candidates.copy()
    n_capped = 0
    if max_share is not None:
        big = Xp.sum(axis=1) > max_share * yp.sum()
        n_capped = int((cand & big).sum())
        cand &= ~big

    t = time.perf_counter()
    xy = case.units[["x", "y"]].to_numpy()
    res = contig.search(Xp, yp, case.graph, cand, objective=objective, area=area, top=top,
                        time_limit=time_limit, dist_to_treated=case.dist, xy=xy, center_spacing=center_spacing,
                        verbose=verbose)
    secs = time.perf_counter() - t
    t = time.perf_counter()
    res_fit = contig.search(Xp, yp, case.graph, cand, objective="fit", area=area, top=10,
                            time_limit=time_limit, xy=xy, center_spacing=center_spacing, verbose=verbose)
    secs_fit = time.perf_counter() - t

    out = {"search": res, "search_fit": res_fit, "seconds": secs, "seconds_fit": secs_fit, "rows": [], "series": {},
           "T0": T0, "T1": T1}

    def add(method, mask=None, k=None, weights=None, extra=None):
        if weights is not None:
            series = weights @ X[cand]
            est = wdd_weighted(T0, T1, X[cand][:, w0].sum(1), X[cand][:, w1].sum(1), weights)
            n_units = int((weights > 1e-3).sum())
            k = 1.0
        else:
            series = X[mask].sum(axis=0)
            if k is None:
                k = series[pre].sum() / max(yp.sum(), 1)
            est = wdd(T0, T1, series[w0].sum(), series[w1].sum(), k=k)
            n_units = int(mask.sum())
        fit = float(res.criteria.fit(series[pre] / k))
        viol = float(res.criteria.violation(series[pre] / k))
        out["series"][method] = series / k
        phi = res.criteria.phi
        row = {"method": method, "k": k, "n_units": n_units, "fit": fit, "viol": viol,
               "C0": series[w0].sum(), "C1": series[w1].sum(),
               "est": est.est, "se": est.se, "low": est.low, "high": est.high, "z": est.z, "pct": est.pct,
               "se_phi": est.se * np.sqrt(phi), "low_phi": est.est - 1.959964 * est.se * np.sqrt(phi),
               "high_phi": est.est + 1.959964 * est.se * np.sqrt(phi)}
        if mask is not None:
            row["area_sqmi"] = area[mask].sum() / 2.59e6
            row["dist_mi"] = float(case.dist[mask].min()) / 5280
        row.update(extra or {})
        out["rows"].append(row)

    def mask_of(nodes):
        m = np.zeros(len(X), dtype=bool)
        m[nodes] = True
        return m

    add("contiguous", mask_of(res.best.nodes), k=1.0, extra={"status": res.best.status})
    add("scan", mask_of(res.scan.nodes), k=1.0)
    add("best fit", mask_of(res_fit.best.nodes), k=1.0)
    if ring is not None:
        add("ring", ring_units(case, *ring, candidates=cand))
    add("city", cand.copy())
    t = time.perf_counter()
    cal = microsynth.calibrate(Xp[cand], yp, case.treated_units.sum())
    add("microsynth", weights=cal.w, extra={"exact": cal.exact, "seconds": time.perf_counter() - t})
    out["calibration"] = cal
    out["tau"] = res.tau
    out["criteria"] = res.criteria
    out["candidates"] = cand
    out["n_capped"] = n_capped
    return out
