"""Step 5: control areas and WDD tests for the two case studies.

RedBird PID (Dallas): services began with the district's first assessment
year, January 1, 2026. Matching uses 84 months (2019-2025); the WDD compares
January-September 2026 with the nine months before.

Operation Restore Roosevelt (Queens): launched October 15, 2024 on Roosevelt
Avenue from 74th to 111th Street. Matching uses 72 months (October 15, 2018
to October 14, 2024); the WDD compares the 20 months after the launch with
the 20 months before. Months run from the 15th to the 14th.

Every method's pool drops blocks holding more than 20% of the treated area's
pre-period crime (case.evaluate's max_share), so no single place dominates a
control area.

Writes results/cases/{case}_*.csv, {case}_meta.json and {case}_areas.gpkg.
"""

import json
import sys
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
out = root / "results" / "cases"
out.mkdir(parents=True, exist_ok=True)

OBJECTIVE = "near"
BUFFER = 1320.0  # quarter mile exclusion buffer (feet)
RING = (BUFFER, 2 * BUFFER)  # donut control: the next quarter mile out
N_ALT = 5  # distinct alternative control areas to report


def load(city):
    units = gpd.read_file(data / f"{city}_units.gpkg")
    adj = sparse.load_npz(data / f"{city}_adj.npz").tocsr()
    crime = pd.read_csv(data / f"{city}_crime.csv.zip", parse_dates=["date"])
    return units, adj, crime


def distinct(sols, max_overlap=0.5, n=N_ALT):
    """The first ``n`` solutions sharing at most ``max_overlap`` of units with any earlier pick."""
    picks = []
    for s in sols:
        S = set(s.nodes.tolist())
        if all(len(S & set(p.nodes.tolist())) / min(len(S), len(p.nodes)) <= max_overlap for p in picks):
            picks.append(s)
        if len(picks) == n:
            break
    return picks


def run(name, c, ring, n_after_months):
    meta = {"name": name, "start": str(c.start.date()), "months": c.months, "n_pre": c.n_pre, "n_post": c.n_post,
            "n_units": len(c.units), "n_candidates": int(c.candidates.sum()),
            "n_treated_units": int(c.treated_units.sum()), "buffer_ft": BUFFER, "ring_ft": list(ring),
            "objective": OBJECTIVE, "types": {}}
    gpd.GeoDataFrame({"part": ["treated", "buffer"]},
                     geometry=[c.treated_geom, c.treated_geom.buffer(BUFFER)], crs=c.units.crs
                     ).to_file(out / f"{name}_areas.gpkg", driver="GPKG")
    c.periods.assign(start=c.periods["start"].dt.date, end=c.periods["end"].dt.date).to_csv(
        out / f"{name}_periods.csv", index=False)
    geoid = c.units["GEOID20"].to_numpy()
    for ctype in ["violent", "property"]:
        print(f"== {name} {ctype}", flush=True)
        r = cs.evaluate(c, ctype, objective=OBJECTIVE, ring=ring)
        rows = pd.DataFrame(r["rows"])
        print(rows[["method", "k", "n_units", "fit", "est", "se", "pct"]].round(2).to_string(), flush=True)
        rows.to_csv(out / f"{name}_{ctype}_wdd.csv", index=False)
        series = pd.DataFrame({"period": c.periods["period"], "treated": c.y[ctype]})
        for m, s in r["series"].items():
            series[m] = s
        series.to_csv(out / f"{name}_{ctype}_series.csv", index=False)

        sel = []
        for m, s in [("contiguous", r["search"].best), ("scan", r["search"].scan), ("best fit", r["search_fit"].best)]:
            sel.append(pd.DataFrame({"method": m, "GEOID20": geoid[s.nodes], "center": geoid[s.center]}))
        pd.concat(sel).to_csv(out / f"{name}_{ctype}_selected.csv", index=False)
        cand = np.flatnonzero(r["candidates"])
        w = r["calibration"].w
        pd.DataFrame({"GEOID20": geoid[cand], "weight": w}).query("weight > 1e-4").to_csv(
            out / f"{name}_{ctype}_microsynth.csv", index=False)

        # Distinct alternatives among the integer program solutions, in rank
        # order (passing areas nearest first, then the rest by violation).
        X, y = c.X[ctype], c.y[ctype]
        n0 = slice(c.n_pre - c.n_post, c.n_pre)
        n1 = slice(c.n_pre, c.n_pre + c.n_post)
        T0, T1 = y[n0].sum(), y[n1].sum()
        ok = [s for s in r["search"].ilp if s.viol <= 1 + 1e-9]
        alts = []
        for i, s in enumerate(distinct(r["search"].ilp)):
            ser = X[s.nodes].sum(axis=0)
            e = wdd(T0, T1, ser[n0].sum(), ser[n1].sum())
            alts.append({"rank": i + 1, "center": geoid[s.center], "n_units": len(s.nodes), "fit": s.cost,
                         "viol": s.viol,
                         "area_sqmi": s.area / 2.59e6, "dist_mi": c.dist[s.nodes].min() / 5280,
                         "est": e.est, "se": e.se, "low": e.low, "high": e.high, "pct": e.pct})
            sel.append(pd.DataFrame({"method": f"alt{i + 1}", "GEOID20": geoid[s.nodes], "center": geoid[s.center]}))
        pd.DataFrame(alts).to_csv(out / f"{name}_{ctype}_alternatives.csv", index=False)

        # Monthly series for the cumulative (CompStat counter) WDD: the whole
        # matching period before, and every full month after the intervention.
        masks = {}
        for m, s in [("contiguous", r["search"].best), ("scan", r["search"].scan)]:
            mk = np.zeros(len(c.units), dtype=bool)
            mk[s.nodes] = True
            masks[m] = mk
        masks["city"] = r["candidates"]
        mon = cs.monthly(c, ctype, c.n_pre * c.months, n_after_months, masks, {"microsynth": w},
                         candidates=r["candidates"])
        mon.assign(start=mon["start"].dt.date).to_csv(out / f"{name}_{ctype}_monthly.csv", index=False)
        pd.concat(sel).to_csv(out / f"{name}_{ctype}_selected.csv", index=False)
        meta["types"][ctype] = {
            "tau": r["tau"], "level_thr": r["criteria"].level, "slope_thr": r["criteria"].slope,
            "phi": r["criteria"].phi,
            "seconds": r["seconds"], "seconds_fit": r["seconds_fit"],
            "search_seconds": r["search"].seconds, "n_windows": r["search"].n_windows,
            "n_feasible_ilp": len(ok), "n_ilp": len(r["search"].ilp),
            "ilp_status": pd.Series([s.status for s in r["search"].ilp]).value_counts().to_dict(),
            "microsynth_exact": bool(r["calibration"].exact), "microsynth_intercept": bool(r["calibration"].intercept),
            "T0": int(T0), "T1": int(T1),
            "n_capped": r["n_capped"], "n_candidates": int(r["candidates"].sum()),
        }
    with open(out / f"{name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)


which = sys.argv[1:] or ["dallas", "nyc"]

if "dallas" in which:
    units, adj, crime = load("dallas")
    pids = gpd.read_file(data / "dallas_pids.gpkg").to_crs(units.crs)
    redbird = pids.loc[pids["Name"] == "RedBird PID"].geometry.union_all()
    others = [g for g in pids.loc[pids["Name"] != "RedBird PID"].geometry]
    c = cs.build_case("dallas", units, adj, crime, redbird, start="2026-01-01", months=1, n_pre=84, n_post=9,
                      buffer=BUFFER, exclude_geoms=others)
    run("dallas", c, RING, n_after_months=9)  # January to September 2026

if "nyc" in which:
    units, adj, crime = load("nyc")
    corridor = roosevelt_corridor(units.crs).buffer(250.0)
    c = cs.build_case("nyc", units, adj, crime, corridor, start="2024-10-15", months=1, n_pre=72, n_post=20,
                      buffer=BUFFER, treated_count_buffer=0.0)
    run("nyc", c, RING, n_after_months=20)  # October 15, 2024 to June 14, 2026
