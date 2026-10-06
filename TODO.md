# TODO / progress log

Status file so a later session can pick up where the last one stopped.
The first unchecked item is where to resume.

## Pipeline

- [x] uv project (`wddcontrol` package under `src/`), kernel `wddcontrol` registered
- [x] `scripts/01_dallas_crime.py` -- Dallas PD violent + property incidents 2019-01-01 to 2026-09-30, PID polygons
- [x] `scripts/02_nyc_crime.py` -- NYPD complaints, Queens, 2018-01-01 to 2026-06-30
- [x] `scripts/03_geography.py` -- 2020 census blocks (land only), rook adjacency, crime -> block
- [x] `scripts/04_simulation.py {smooth,local} 0 49` -- Monte Carlo, 50 worlds per scenario (resumable; ~45 s per world; extend with a larger last seed)
- [x] `scripts/05_case_studies.py` -- RedBird PID (Dallas) and Operation Restore Roosevelt (Queens)
- [x] `scripts/06_nocap.py` -- Queens property without the single-place cap
- [x] `scripts/07_cap_sensitivity.py` -- Queens property with 10% and 5% caps
- [x] paper.qmd written and rendered (pdf/docx/md); abstract numbers are hard-coded, update if results change
- [x] GitHub repo apwheele/WDDControl, public since 2026-10-06 after Andrew's review

## Design decisions (and why)

- Matching is monthly (Andrew's request), k = 1.
- A control "passes" when four gap summaries are within noise: monthly
  L1 fit, 6-month block fit, level, trend (contig.Criteria). Noise is
  Poisson scaled by phi, the treated series' Pearson dispersion around a
  7-month moving average (Queens series have phi 1.5-1.7; with pure
  Poisson nothing passed for Queens property and the pick was unstable).
- Best fit (min L1) overfits noise: worst or near-worst RMSE in sims.
- Among passing areas: nearest window to the treated area; within a window
  min sum of network distances (p-median compactness), then the best 8
  passing areas are re-solved with perimeter added (fills holes).
- Rook adjacency (shared boundary >= 20 ft): queen let corner-touching
  blocks form checkerboard chains.
- Closer-neighbor ordering constraint keeps each window ILP small.
- Single-place cap: no block with > 20% of treated pre-period crime in any
  method's pool (chain and department store blocks in Queens whose
  shoplifting complaints collapsed after 2024 wrecked uncapped controls).
- Microsynth replication: raking via Newton on the dual (matches cvxpy);
  with the intercept (sum w = n_treated) exact calibration is infeasible in
  Queens and the sims, so it falls back to outcomes only.
- Case-study centers thinned to one per 800 ft cell (speed).
- No holes (Andrew's request after first review): ILP rows forbid one-unit
  holes (z_i >= sum of neighbors - (deg - 1)) and enclosing an ineligible
  unit; larger enclosed pockets of candidate units are filled
  (contig.fill_holes) and areas are ranked on filled versions. Units on
  the region edge (geo.edge_units) cannot be enclosed.
- Synthetic worlds are simulated on 40x40 and analyzed on 20x20 coarse
  cells (Andrew asked for coarser units); 2x2 treated, 1-cell buffer.
- Roosevelt date checked against the Oct 15, 2024 NYC press release; the
  corridor's violent crime peaked June-July 2024 and fell before launch
  (city says enforcement was ongoing over the prior year).
- Cumulative chart shows only the contiguous-control running WDD with its
  95% band (Andrew: one error area, not two).
- ILP time limits make reruns slightly machine dependent.
- Dropbox syncing .venv slowed runs; Andrew paused Dropbox 2026-10-06.
