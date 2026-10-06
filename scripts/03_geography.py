"""Step 3: census block units, adjacency, and block assignment of incidents.

Dallas: 2020 blocks with land area inside the city of Dallas (NAD83 Texas
North Central, feet). Queens: 2020 blocks with land area in Queens County
(NAD83 New York Long Island, feet).

Adjacency is rook contiguity (a shared boundary of at least 20 feet), so blocks
that only touch at a corner are not neighbors.

Writes data/{city}_units.gpkg, data/{city}_adj.npz, and adds a ``unit``
column (block index, -1 if outside) to data/{city}_crime.csv.zip.
"""

from pathlib import Path

import pandas as pd
from scipy import sparse

from wddcontrol import geo

data = Path(__file__).resolve().parents[1] / "data"

SETUP = {
    "dallas": {"crs": 2276, "state": "TX"},
    "nyc": {"crs": 2263, "state": "NY"},
}

for city, cfg in SETUP.items():
    if city == "dallas":
        boundary = geo.place_boundary("Dallas", "TX", cfg["crs"])
        counties = geo.overlapping_counties(boundary, "TX")
    else:
        boundary = geo.county_boundary("NY", "081", cfg["crs"])  # Queens
        counties = ["081"]
    blocks = geo.census_blocks(boundary, cfg["state"], counties)
    adj = geo.adjacency(blocks)
    n_comp = sparse.csgraph.connected_components(adj)[0]
    print(f"{city}: {len(blocks):,} blocks, {adj.nnz // 2:,} adjacent pairs, {n_comp} connected pieces")
    boundary.to_file(data / f"{city}_boundary.gpkg", driver="GPKG")
    blocks.to_file(data / f"{city}_units.gpkg", driver="GPKG")
    sparse.save_npz(data / f"{city}_adj.npz", adj)

    crime = pd.read_csv(data / f"{city}_crime.csv.zip", parse_dates=["date"])
    crime["unit"] = geo.assign_points(crime, blocks)
    print(f"{city}: {len(crime):,} incidents, {(crime['unit'] >= 0).mean():.1%} assigned to a block")
    crime.to_csv(data / f"{city}_crime.csv.zip", index=False, compression="zip")
