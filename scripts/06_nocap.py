"""Step 6: Queens property crime without the single-place cap.

Reruns the Operation Restore Roosevelt property crime search with every
block allowed (case.evaluate max_share=None), to show what the cap guards
against. Writes results/cases/nocap/nyc_property_{wdd,monthly,blocks}.csv:
the WDD table, monthly series, and the selected blocks' pre-period and
post-period crime with their premise mix.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse

from wddcontrol import case as cs
from wddcontrol import geo
from wddcontrol.nyc import roosevelt_corridor

root = Path(__file__).resolve().parents[1]
data = root / "data"
out = root / "results" / "cases" / "nocap"
out.mkdir(parents=True, exist_ok=True)

units = gpd.read_file(data / "nyc_units.gpkg")
adj = sparse.load_npz(data / "nyc_adj.npz").tocsr()
crime = pd.read_csv(data / "nyc_crime.csv.zip", parse_dates=["date"])
corridor = roosevelt_corridor(units.crs).buffer(250.0)
c = cs.build_case("nyc", units, adj, crime, corridor, start="2024-10-15", months=1, n_pre=72, n_post=20,
                  buffer=1320.0, treated_count_buffer=0.0)
r = cs.evaluate(c, "property", ring=(1320.0, 2640.0), max_share=None)
pd.DataFrame(r["rows"]).to_csv(out / "nyc_property_wdd.csv", index=False)
sel = r["search"].best.nodes
mask = np.zeros(len(units), dtype=bool)
mask[sel] = True
mon = cs.monthly(c, "property", c.n_pre, 20, {"contiguous": mask}, {})
mon.assign(start=mon["start"].dt.date).to_csv(out / "nyc_property_monthly.csv", index=False)

# Each selected block's crime in the two years before and the 20 months after.
prop = crime[(crime["crime_type"] == "property") & crime["unit"].isin(sel)].copy()
month = geo.period_index(prop["date"], c.start, 1)
prop["window"] = np.where((month >= -24) & (month < 0), "pre", np.where((month >= 0) & (month < 20), "post", ""))
tab = prop[prop["window"] != ""].groupby(["unit", "window"]).size().unstack(fill_value=0)
top = prop[prop["window"] == "pre"].groupby("unit")["prem_typ_desc"].agg(lambda s: s.value_counts().index[0])
tab = tab.join(top.rename("top_premise"))
tab["GEOID20"] = units["GEOID20"].to_numpy()[tab.index]
tab["share_pre"] = tab["pre"] / tab["pre"].sum()
tab.sort_values("pre", ascending=False).to_csv(out / "nyc_property_blocks.csv")
print(pd.DataFrame(r["rows"])[["method", "n_units", "fit", "viol", "est", "se", "pct"]].round(2))
print(tab.sort_values("pre", ascending=False).head())
