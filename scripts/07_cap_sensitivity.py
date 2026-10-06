"""Step 7: Queens property crime with stricter single-place caps.

Reruns the Operation Restore Roosevelt property crime search with no block
allowed more than 10% or 5% of the treated area's pre-period crime, and
records the WDD for the five best distinct control areas under each cap.
Writes results/cases/cap_sensitivity.csv.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse

from wddcontrol import case as cs
from wddcontrol.nyc import roosevelt_corridor
from wddcontrol.wdd import wdd

root = Path(__file__).resolve().parents[1]
data = root / "data"

units = gpd.read_file(data / "nyc_units.gpkg")
adj = sparse.load_npz(data / "nyc_adj.npz").tocsr()
crime = pd.read_csv(data / "nyc_crime.csv.zip", parse_dates=["date"])
corridor = roosevelt_corridor(units.crs).buffer(250.0)
c = cs.build_case("nyc", units, adj, crime, corridor, start="2024-10-15", months=1, n_pre=72, n_post=20,
                  buffer=1320.0, treated_count_buffer=0.0)
X, y = c.X["property"], c.y["property"]
n0, n1 = slice(c.n_pre - c.n_post, c.n_pre), slice(c.n_pre, c.n_pre + c.n_post)
T0, T1 = y[n0].sum(), y[n1].sum()


def distinct(sols, max_overlap=0.5, n=5):
    picks = []
    for s in sols:
        S = set(s.nodes.tolist())
        if all(len(S & set(p.nodes.tolist())) / min(len(S), len(p.nodes)) <= max_overlap for p in picks):
            picks.append(s)
        if len(picks) == n:
            break
    return picks


rows = []
for cap in [0.10, 0.05]:
    r = cs.evaluate(c, "property", max_share=cap)
    for i, s in enumerate(distinct(r["search"].ilp)):
        ser = X[s.nodes].sum(axis=0)
        e = wdd(T0, T1, ser[n0].sum(), ser[n1].sum())
        rows.append({"cap": cap, "rank": i + 1, "n_units": len(s.nodes), "viol": s.viol,
                     "dist_mi": c.dist[s.nodes].min() / 5280, "est": e.est, "se": e.se,
                     "n_capped": r["n_capped"]})
    print(pd.DataFrame(rows).round(2).tail(5))
pd.DataFrame(rows).to_csv(root / "results" / "cases" / "cap_sensitivity.csv", index=False)
