# WDDControl

How should a crime analyst draw the control area for a weighted
displacement difference (WDD) test? This repository implements a method that
picks a single, contiguous geographic area (no weights) whose crime before
the intervention tracks the treated area's, period by period, in level and
trend, within what Poisson noise allows. The search is a network scan over
every possible center combined with a small integer program inside each
scan window; a constraint that each selected unit borders a selected unit
closer to the window's center keeps the area connected and compact, and
keeps each program small enough to solve in about a second. Control areas
have no holes: the program forbids one-unit holes and any larger enclosed
pocket is filled before an area is checked and ranked.

The paper is `paper.pdf` (also built as `paper.docx` and `paper.md`), built
from `paper.qmd` and `references.bib`. It compares the method with common
informal control areas (a surrounding ring, the whole city), a pure scan,
the best-fitting contiguous area, and a Python replication of microsynth, on
synthetic grids and two case studies: the RedBird public improvement
district in Dallas, Texas, and Operation Restore Roosevelt in Queens, New
York.

## Reproducing

This project uses [uv](https://docs.astral.sh/uv/) for the Python environment
and [Quarto](https://quarto.org/) to render the paper.

```bash
uv sync
uv run python -m ipykernel install --user --name wddcontrol --display-name "Python (wddcontrol)"

uv run python scripts/01_dallas_crime.py     # Dallas PD incidents + PID boundaries -> data/
uv run python scripts/02_nyc_crime.py        # NYPD complaints (Queens) -> data/
uv run python scripts/03_geography.py        # census blocks, adjacency, crime -> block
uv run python scripts/04_simulation.py smooth 0 49  # Monte Carlo replications -> results/sim/ (resumable)
uv run python scripts/04_simulation.py local 0 49
uv run python scripts/05_case_studies.py     # both case studies -> results/cases/
uv run python scripts/06_nocap.py            # Queens property without the single-place cap
uv run python scripts/07_cap_sensitivity.py     # Queens property with 10% and 5% caps
quarto render paper.qmd --to all
uv run pytest                                # unit tests
```

The paper uses 50 simulated worlds per scenario; the script is resumable, so
running it again with a larger last seed adds more. Each world takes about a minute and each case study search two to
four minutes on a four-core desktop. The integer programs run under time
limits, so a rerun on a different machine can pick a slightly different
control area where several are nearly tied. The paper only reads
`results/`, so rendering does not rerun the searches.
The processed data in `data/` are committed, so steps 1 to 3 can be skipped;
the Dallas and New York open data portals update continuously, so
re-downloading may give slightly different counts.

## Using the method

The search works on small geographic units (here census blocks). Every
input is an array with one entry per unit, in the same row order as the
units' GeoDataFrame, so unit `i` is row `i` of `units`, row `i` of `X`,
and row and column `i` of the adjacency matrix.

| Input | Type and shape | What it is |
|---|---|---|
| `X` | array, (n_units, P) | Crime counts for every unit in each of the P pre-period months |
| `y` | array, (P,) | The treated area's counts in the same months |
| `graph` | sparse matrix, (n_units, n_units) | Which units share a border, weighted by the distance between them: `contig.distance_graph(adj, xy)` |
| `candidates` | boolean array, (n_units,) | Units the control area may use (outside the treated area's buffer, not in other treated areas) |
| `dist_to_treated` | array, (n_units,) | Each unit's distance from the treated area; the nearest passing area is chosen |
| `area` | array, (n_units,) | Optional. Land area of each unit, for reporting and breaking ties |
| `xy`, `center_spacing` | array (n_units, 2), number | Optional. Unit coordinates and a grid size; keeps one scan center per grid cell, which speeds up the search |
| `edge` | boolean array, (n_units,) | Optional. Units on the edge of the study region (`geo.edge_units(units)`), which cannot be enclosed by a control area |

`graph` is built from two things: `adj`, a 0/1 sparse matrix marking which
units share a border (`geo.adjacency(units)`), and `xy`, a point inside each
unit. Distances are in the units of the map projection (feet here).

### Example: the RedBird PID in Dallas

This runs on the data committed in `data/` and takes about three minutes.
It finds the violent crime control area used in the paper: 73 blocks south
of the district that pass all four checks (`res.best.viol` 1.00), with a
WDD of -6 (SE 10.8).

```python
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse

from wddcontrol import contig, geo
from wddcontrol.wdd import cumulative_wdd, wdd

# 1. Units: census blocks in a projected CRS (feet), and which blocks touch
units = gpd.read_file("data/dallas_units.gpkg")    # one row per block: GEOID20, ALAND20, x, y, geometry
adj = sparse.load_npz("data/dallas_adj.npz")       # (n_units, n_units) 0/1 rook adjacency, rows in units' order

# 2. Incidents: a date and the row of `units` each one falls in (-1 if none)
crime = pd.read_csv("data/dallas_crime.csv.zip", parse_dates=["date"])
crime = crime[crime["crime_type"] == "violent"]

# 3. Treated area: a shapely geometry in the units' CRS
pids = gpd.read_file("data/dallas_pids.gpkg").to_crs(units.crs)
treated = pids.loc[pids["Name"] == "RedBird PID"].geometry.union_all()

# 4. Monthly counts: 84 months before January 1, 2026 to match on, 9 after for the WDD
n_pre, n_post = 84, 9
month = geo.period_index(crime["date"], pd.Timestamp("2026-01-01"), months=1)   # 0 = Jan 2026, -1 = Dec 2025
X = geo.unit_counts(crime["unit"].to_numpy(), month, len(units), -n_pre, n_post - 1)  # (n_units, 93)
in_treated = geo.points_in(crime, treated.buffer(100), units.crs)   # 100 ft catches boundary streets
keep = in_treated & (month >= -n_pre) & (month < n_post)
y = np.bincount(month[keep] + n_pre, minlength=n_pre + n_post)      # (93,)
X_pre, y_pre = X[:, :n_pre], y[:n_pre]

# 5. Candidate pool: which blocks the control area may use
dist = units.distance(treated).to_numpy()            # feet from the treated area
candidates = dist > 1320                             # quarter-mile buffer
for g in pids.loc[pids["Name"] != "RedBird PID"].geometry:
    candidates &= ~units.intersects(g).to_numpy()    # other improvement districts
candidates &= X_pre.sum(axis=1) <= 0.2 * y_pre.sum() # no block with > 20% of the treated area's crime

# 6. Search
xy = units[["x", "y"]].to_numpy()
graph = contig.distance_graph(adj, xy)               # adjacency weighted by distance between blocks
res = contig.search(X_pre, y_pre, graph, candidates, dist_to_treated=dist,
                    area=units["ALAND20"].to_numpy(), xy=xy, center_spacing=800,
                    edge=geo.edge_units(units), verbose=True)

control = units.iloc[res.best.nodes]                 # the control area's blocks, a GeoDataFrame
control.dissolve().to_file("redbird_control.gpkg")

# 7. WDD, the 9 months after against the 9 before
c = X[res.best.nodes].sum(axis=0)                    # the control area's monthly counts
before, after = slice(n_pre - 9, n_pre), slice(n_pre, n_pre + n_post)
print(wdd(y[before].sum(), y[after].sum(), c[before].sum(), c[after].sum()))
running = cumulative_wdd(y[before], c[before], y[after], c[after])  # month-by-month running WDD
```

### What comes back

`contig.search` returns a `SearchResult`:

| Field | What it is |
|---|---|
| `res.best.nodes` | Row positions in `units` of the control area's blocks |
| `res.best.viol` | The largest of the four pre-period gap checks relative to its noise threshold; at most 1 means the area passes |
| `res.best.cost` | Monthly fit: the sum over pre-period months of the absolute difference between the control's counts and the treated area's |
| `res.best.area`, `res.best.center` | Summed `area` of the blocks, and the row of the scan window's center |
| `res.scan` | The best pure scan area (a network circle), for comparison |
| `res.ilp` | Every window's integer program solution, best first: the equally good alternatives |
| `res.criteria` | The thresholds; `res.criteria.phi` is the estimated dispersion, and `res.criteria.violation(series)` checks any other control series |

Settings you may want to change: `k` (the control tracks `k` times the
treated area's crime, default 1; pass the same `k` to `wdd`), `dispersion`
(default estimated from `y`; 1 gives pure Poisson thresholds), `tau_mult`
and `z` (loosen or tighten the fit and level/trend thresholds), and
`scales` (block lengths for the medium-term check, default `(6,)` months).

### Your own data

You need polygons in a projected CRS, with `x` and `y` columns for a point
inside each, and incidents with a date and `lat`/`lon`:

```python
units = geo.census_blocks(boundary, "TX", counties)  # or any polygons with x, y columns
adj = geo.adjacency(units)                           # rook: a shared border of at least 20 CRS units
crime["unit"] = geo.assign_points(crime, units)      # row of `units` holding each incident, -1 if none
```

Then follow steps 3 to 7 above. `case.build_case` and `case.evaluate`
bundle steps 4 to 7 and also run the comparison controls (ring, city,
best fit, microsynth); `scripts/05_case_studies.py` uses them.

## Layout

- `paper.qmd`, `references.bib` -- paper source and bibliography.
- `src/wddcontrol/` -- the package:
  - `contig.py` -- pass/fail criteria, network scan windows, greedy and integer program (HiGHS).
  - `microsynth.py` -- raking calibration weights (cvxpy), a Python version of microsynth's weighting.
  - `wdd.py` -- the WDD estimate and Poisson standard error, for areas and weighted controls.
  - `sim.py` -- synthetic grid worlds.
  - `geo.py` -- census blocks, adjacency, assigning incidents to blocks and periods.
  - `case.py` -- case study setup and evaluation.
  - `dallas.py`, `nyc.py` -- open data downloads and the Roosevelt Avenue corridor.
- `scripts/` -- the numbered pipeline above.
- `data/` -- processed incidents, census blocks, adjacency matrices, PID boundaries.
- `results/` -- simulation rows and case study outputs read by the paper.
- `filters/` -- pandoc Lua filters for the Markdown and Word builds.

## License

MIT, see `LICENSE`.
