"""Step 4: Monte Carlo comparison of control areas on synthetic grids.

Each replication draws a 40 x 40 grid world (sim.simulate) with a 4 x 4
treated hot spot, a 2-cell exclusion buffer, 60 pre-period months and 12 post
months, and a 20% reduction in the treated cells after the intervention, in
one of two scenarios ("smooth" or "local" trends), and sums it into a 20 x 20
grid of coarse cells (2 x 2 treated, 1-cell buffer). Control areas are chosen
from the 60 pre-period months; the WDD compares the last 12 pre months with
the 12 post months.

Usage: uv run python scripts/04_simulation.py SCENARIO FIRST LAST
Writes one CSV per replication to results/sim/rows/ (skips finished ones),
and replication EXAMPLE_SEED's world and selections to
results/sim/example_{scenario}.npz.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wddcontrol import contig, microsynth, sim
from wddcontrol.wdd import wdd, wdd_weighted

root = Path(__file__).resolve().parents[1]
out = root / "results" / "sim" / "rows"
out.mkdir(parents=True, exist_ok=True)
EXAMPLE_SEED = 0
TOP, TIME_LIMIT = 20, 1.0
COARSE = 2


def run(scenario: str, seed: int, save_example: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = sim.simulate(rng, scenario=scenario, coarse=COARSE)
    n = w.nr * w.nc
    X = w.counts[:, :w.n_pre]
    y = X[w.treated].sum(axis=0)
    g = contig.distance_graph(w.adj, w.xy)
    cand = ~w.excluded
    c0 = w.counts[:, w.n_pre - 12:w.n_pre].sum(axis=1)
    c1 = w.counts[:, w.n_pre:].sum(axis=1)
    l0 = w.lam[:, w.n_pre - 12:w.n_pre].sum(axis=1)
    l1 = w.lam[:, w.n_pre:].sum(axis=1)
    T0, T1 = c0[w.treated].sum(), c1[w.treated].sum()
    truth = w.effect * l1[w.treated].sum()
    dT = l1[w.treated].sum() - l0[w.treated].sum()  # expected change without the intervention
    # Chebyshev and Euclidean distance (in cells) from the treated block
    rr, cc = np.divmod(np.arange(n), w.nc)
    tr = np.argwhere(w.treated.reshape(w.nr, w.nc))
    (r0, q0), (r1, q1) = tr.min(axis=0), tr.max(axis=0)
    dr = np.maximum(np.maximum(r0 - rr, rr - r1), 0)
    dc = np.maximum(np.maximum(q0 - cc, cc - q1), 0)
    cheb = np.maximum(dr, dc)
    dist = np.hypot(dr, dc)
    crit = contig.Criteria.from_series(y)

    rows, example = [], {}

    def add(method, mask=None, k=None, weights=None, secs=np.nan):
        if weights is not None:
            e = wdd_weighted(T0, T1, c0[cand], c1[cand], weights)
            series, k = weights @ X[cand], 1.0
            dC = weights @ (l1[cand] - l0[cand])
            size, d = int((weights > 1e-3).sum()), np.nan
            example[method] = np.zeros(n)
            example[method][cand] = weights
        else:
            series = X[mask].sum(axis=0)
            k = series.sum() / y.sum() if k is None else k
            e = wdd(T0, T1, c0[mask].sum(), c1[mask].sum(), k=k)
            dC = (l1[mask].sum() - l0[mask].sum()) / k
            size, d = int(mask.sum()), float(dist[mask].min())
            example[method] = mask.astype(float)
        rows.append({"scenario": scenario, "seed": seed, "method": method, "est": e.est, "se": e.se, "low": e.low,
                     "high": e.high, "truth": truth, "trend_err": dT - dC, "k": k, "phi": crit.phi,
                     "fit": float(crit.fit(series / k)), "viol": float(crit.violation(series / k)),
                     "cells": size, "dist": d, "seconds": secs, "T0": T0, "T1": T1})

    add("ring", cand & (cheb == 2))  # the band of cells just outside the one-cell buffer
    add("city", cand.copy())
    for obj in ["near", "fit"]:
        t = time.perf_counter()
        res = contig.search(X, y, g, cand, objective=obj, top=TOP, time_limit=TIME_LIMIT, refine=1,
                            refine_time=2 * TIME_LIMIT, dist_to_treated=dist, edge=sim.grid_edge(w.nr, w.nc))
        secs = time.perf_counter() - t
        sels = [("scan", res.scan), ("contiguous", res.best)] if obj == "near" else [("best fit", res.best)]
        for name, s in sels:
            m = np.zeros(n, dtype=bool)
            m[s.nodes] = True
            add(name, m, k=1.0, secs=secs)
    t = time.perf_counter()
    cal = microsynth.calibrate(X[cand], y, w.treated.sum())
    add("microsynth", weights=cal.w, secs=time.perf_counter() - t)
    if save_example:
        np.savez_compressed(root / "results" / "sim" / f"example_{scenario}.npz", lam=w.lam, counts=w.counts,
                            treated=w.treated, excluded=w.excluded, nr=w.nr, nc=w.nc, n_pre=w.n_pre,
                            **{f"sel_{k}": v for k, v in example.items()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    scenario, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    for seed in range(first, last + 1):
        f = out / f"{scenario}_{seed:04d}.csv"
        if f.exists():
            continue
        t = time.perf_counter()
        df = run(scenario, seed, save_example=(seed == EXAMPLE_SEED))
        df.to_csv(f, index=False)
        print(f"{scenario} seed {seed} done in {time.perf_counter() - t:.0f}s", flush=True)
