# WDDControl

How should a crime analyst draw the control area for a weighted
displacement difference (WDD) test? This repository implements a method that
picks a single, contiguous geographic area (no weights) whose crime before
the intervention tracks the treated area's, period by period, in level and
trend, within what Poisson noise allows. The search is a network scan over
every possible center combined with a small integer program inside each
scan window; a constraint that each selected unit borders a selected unit
closer to the window's center keeps the area connected and compact, and
keeps each program small enough to solve in about a second.

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

```python
from wddcontrol import contig

graph = contig.distance_graph(adjacency, unit_xy)    # adjacency weighted by distance
res = contig.search(
    X,                    # (units, periods) pre-period counts
    y,                    # (periods,) treated area counts
    graph,
    candidates,           # boolean mask: units outside the treated area's buffer
    dist_to_treated=d,    # distance of each unit from the treated area
)
res.best.nodes            # indices of the units in the control area
res.ilp                   # all window solutions, best first (alternatives)
```

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
