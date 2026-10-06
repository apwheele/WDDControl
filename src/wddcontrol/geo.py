"""Census block units, block adjacency, and crime counts by unit and period."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pygris
import shapely
from scipy import sparse


def place_boundary(name: str, state: str, crs, year: int = 2020) -> gpd.GeoDataFrame:
    places = pygris.places(state, year=year, cache=True)
    return places.loc[places["NAME"] == name, ["GEOID", "NAME", "geometry"]].to_crs(crs)


def county_boundary(state: str, county: str, crs, year: int = 2020) -> gpd.GeoDataFrame:
    counties = pygris.counties(state, year=year, cache=True)
    return counties.loc[counties["COUNTYFP"] == county, ["GEOID", "NAME", "geometry"]].to_crs(crs)


def overlapping_counties(boundary: gpd.GeoDataFrame, state: str, year: int = 2020) -> list[str]:
    counties = pygris.counties(state, year=year, cache=True).to_crs(boundary.crs)
    geom = boundary.geometry.union_all()
    return sorted(counties.loc[counties.intersects(geom), "COUNTYFP"].tolist())


def census_blocks(boundary: gpd.GeoDataFrame, state: str, counties: list[str], year: int = 2020) -> gpd.GeoDataFrame:
    """2020 census blocks with land area inside ``boundary``.

    Blocks nest within 2020 place and county boundaries, so a representative
    point test selects them without sliver overlaps. Water-only blocks are
    dropped, so a control area cannot be connected across a lake or river.
    """
    parts = [pygris.blocks(state, c, year=year, cache=True) for c in counties]
    blocks = pd.concat(parts, ignore_index=True).to_crs(boundary.crs)
    blocks = blocks[blocks["ALAND20"] > 0]
    rep = gpd.GeoSeries(blocks.representative_point(), crs=boundary.crs)
    blocks = blocks[rep.within(boundary.geometry.union_all()).to_numpy()]
    blocks = blocks[["GEOID20", "ALAND20", "POP20", "geometry"]].reset_index(drop=True)
    rep = blocks.representative_point()
    blocks["x"] = rep.x
    blocks["y"] = rep.y
    return blocks


def adjacency(units: gpd.GeoDataFrame, tol: float = 1.0, min_shared: float = 20.0) -> sparse.csr_matrix:
    """Symmetric 0/1 rook adjacency: polygons sharing at least ``min_shared`` of boundary.

    Candidate pairs are polygons within ``tol`` of each other; the shared
    boundary is the length of one polygon's boundary inside the other's
    ``tol`` buffer. Blocks that only touch at a corner (diagonally across an
    intersection) are not neighbors.
    """
    buf = gpd.GeoDataFrame(geometry=units.buffer(tol), crs=units.crs)
    pairs = gpd.sjoin(buf, units[["geometry"]], how="inner", predicate="intersects")
    i = pairs.index.to_numpy()
    j = pairs["index_right"].to_numpy()
    keep = i < j
    i, j = i[keep], j[keep]
    shared = shapely.length(shapely.intersection(units.boundary.values[j], buf.geometry.values[i]))
    ok = shared >= min_shared
    n = len(units)
    mat = sparse.coo_matrix((np.ones(ok.sum(), dtype=np.int8), (i[ok], j[ok])), shape=(n, n)).tocsr()
    mat = ((mat + mat.T) > 0).astype(np.int8)
    return mat


def edge_units(units: gpd.GeoDataFrame, tol: float = 1.0) -> np.ndarray:
    """Units touching the boundary of the region they cover (outer edge, water, enclaves)."""
    region = shapely.union_all(units.geometry.values)
    line = shapely.buffer(shapely.boundary(region), tol)
    return shapely.intersects(units.geometry.values, line)


def assign_points(crime: pd.DataFrame, units: gpd.GeoDataFrame, max_dist: float = 300.0) -> np.ndarray:
    """Index of the unit containing each incident (nearest within ``max_dist`` if on no unit), -1 if none."""
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(crime["lon"], crime["lat"]), crs=4326).to_crs(units.crs)
    out = np.full(len(pts), -1, dtype=np.int64)
    inside = gpd.sjoin(pts, units[["geometry"]], how="inner", predicate="within")
    inside = inside[~inside.index.duplicated(keep="first")]
    out[inside.index.to_numpy()] = inside["index_right"].to_numpy()
    miss = np.flatnonzero(out < 0)
    if len(miss):
        near = gpd.sjoin_nearest(pts.iloc[miss], units[["geometry"]], how="inner", max_distance=max_dist)
        near = near[~near.index.duplicated(keep="first")]
        out[near.index.to_numpy()] = near["index_right"].to_numpy()
    return out


def points_in(crime: pd.DataFrame, geom, crs) -> np.ndarray:
    """Boolean mask of incidents falling inside ``geom`` (a shapely geometry in ``crs``)."""
    pts = gpd.GeoSeries(gpd.points_from_xy(crime["lon"], crime["lat"]), crs=4326).to_crs(crs)
    return shapely.contains(geom, pts.values) | shapely.touches(geom, pts.values)


def period_index(dates: pd.Series, start: pd.Timestamp, months: int) -> np.ndarray:
    """Period of each date relative to ``start``, in blocks of ``months``.

    Period 0 starts at ``start``; period -1 is the ``months`` before it, etc.
    Months are counted from ``start``'s day of month, so a mid-month start
    gives periods running from the 15th to the 14th.
    """
    d = pd.to_datetime(dates)
    off = (d.dt.year - start.year) * 12 + (d.dt.month - start.month) - (d.dt.day < start.day).astype(int)
    return np.floor_divide(off.to_numpy(), months)


def unit_counts(unit: np.ndarray, period: np.ndarray, n_units: int, first: int, last: int) -> np.ndarray:
    """(n_units, n_periods) counts for periods ``first..last`` inclusive."""
    keep = (unit >= 0) & (period >= first) & (period <= last)
    n_per = last - first + 1
    flat = unit[keep] * n_per + (period[keep] - first)
    return np.bincount(flat, minlength=n_units * n_per).reshape(n_units, n_per)
