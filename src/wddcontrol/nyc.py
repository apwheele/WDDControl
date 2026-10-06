"""Download NYPD complaints for Queens, split into violent and property crime."""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

# NYPD Complaint Data Historic (through the last full year) and Current (YTD).
HISTORIC_URL = "https://data.cityofnewyork.us/resource/qgea-i56i.json"
CURRENT_URL = "https://data.cityofnewyork.us/resource/5uac-w243.json"

# Offense description -> crime type. Violent and property crime follow the
# Dallas definitions: murder, robbery and felony (aggravated) assault; burglary,
# larceny and motor vehicle theft. Rape is left out, since most rape complaints
# are released without coordinates.
CRIME_TYPES = {
    "MURDER & NON-NEGL. MANSLAUGHTER": "violent",
    "ROBBERY": "violent",
    "FELONY ASSAULT": "violent",
    "BURGLARY": "property",
    "GRAND LARCENY": "property",
    "PETIT LARCENY": "property",
    "GRAND LARCENY OF MOTOR VEHICLE": "property",
}


def _soql_list(values):
    return ",".join("'" + v.replace("'", "''") + "'" for v in values)


def _download(url: str, start: str, end: str, boro: str, timeout: int, page: int) -> pd.DataFrame:
    where = (
        f"cmplnt_fr_dt >= '{start}T00:00:00' AND cmplnt_fr_dt < '{end}T00:00:00'"
        f" AND boro_nm = '{boro}' AND ofns_desc IN ({_soql_list(CRIME_TYPES)})"
    )
    select = "cmplnt_num,cmplnt_fr_dt,ofns_desc,law_cat_cd,prem_typ_desc,latitude,longitude"
    parts, offset = [], 0
    while True:
        res = requests.get(
            url,
            params={"$select": select, "$where": where, "$limit": page,
                    "$offset": offset, "$order": "cmplnt_num"},
            timeout=timeout,
        )
        res.raise_for_status()
        rows = res.json()
        parts.append(pd.DataFrame(rows))
        if len(rows) < page:
            break
        offset += page
    return pd.concat(parts, ignore_index=True)


def download_complaints(start: str, end: str, split: str = "2026-01-01", boro: str = "QUEENS",
                        timeout: int = 900, page: int = 200_000) -> pd.DataFrame:
    """Complaints with ``start <= cmplnt_fr_dt < end`` in one borough.

    Occurrence dates before ``split`` come from the historic file, later ones
    from the year-to-date file.
    """
    hist = _download(HISTORIC_URL, start, split, boro, timeout, page)
    curr = _download(CURRENT_URL, split, end, boro, timeout, page)
    return pd.concat([hist, curr], ignore_index=True)


def clean_complaints(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per complaint with crime type, ungeocoded rows dropped."""
    df = raw.drop_duplicates("cmplnt_num").copy()
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["date"] = pd.to_datetime(df["cmplnt_fr_dt"].str[:10])
    df["crime_type"] = df["ofns_desc"].map(CRIME_TYPES)
    df = df[df["lat"].notna() & (df["lat"] != 0)]
    keep = ["cmplnt_num", "date", "ofns_desc", "law_cat_cd", "crime_type", "prem_typ_desc", "lat", "lon"]
    return df[keep].sort_values("date").reset_index(drop=True)


def roosevelt_corridor(crs=2263, west: str = "74th St", east: str = "111th St", year: int = 2024):
    """Centerline of Roosevelt Avenue between two cross streets (TIGER roads, Queens).

    Roosevelt Avenue is clipped to the slab between the perpendiculars
    through its intersections with ``west`` and ``east``.
    """
    import pygris
    import shapely

    roads = pygris.roads("NY", "081", year=year, cache=True).to_crs(crs)
    rv = shapely.line_merge(roads.loc[roads["FULLNAME"] == "Roosevelt Ave"].geometry.union_all())
    p0 = rv.intersection(roads.loc[roads["FULLNAME"] == west].geometry.union_all())
    p1 = rv.intersection(roads.loc[roads["FULLNAME"] == east].geometry.union_all())
    a = shapely.get_coordinates(p0).mean(axis=0)
    b = shapely.get_coordinates(p1).mean(axis=0)
    u = (b - a) / np.hypot(*(b - a))
    v = np.array([-u[1], u[0]])
    w = 3000.0
    slab = shapely.Polygon([a + w * v, b + w * v, b - w * v, a - w * v])
    return shapely.line_merge(rv.intersection(slab))
