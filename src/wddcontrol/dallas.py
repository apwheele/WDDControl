"""Download Dallas Police Department incidents and the RedBird PID boundary.

The Socrata download follows the one used in
https://github.com/apwheele/FairCameraSiting (``faircam/crime.py``), here
pulling all premises and splitting incidents into violent and property crime.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import requests

SOCRATA_URL = "https://www.dallasopendata.com/resource/qv6i-rri7.json"

# City of Dallas Office of Economic Development PID boundaries (2026 tax rolls).
PID_URL = (
    "https://services2.arcgis.com/rwnOSbfKSwyTBcwN/arcgis/rest/services/"
    "Proposed_2026___City_of_Dallas_Public_Improvement_Districts_WFL1/FeatureServer/2/query"
)

# Offense -> crime type. Order is the seriousness ranking used to keep one
# offense per incident: an incident with a robbery and a theft is violent.
# Family-violence aggravated assaults are almost entirely withheld from the
# public file, so violent crime here is mostly non-family violence.
CRIME_TYPES = {
    "MURDER & NONNEGLIGENT MANSLAUGHTER": "violent",
    "AGG ASSAULT - NFV": "violent",
    "AGG ASSAULT - FV": "violent",
    "ROBBERY-INDIVIDUAL": "violent",
    "ROBBERY-BUSINESS": "violent",
    "SHOPLIFT CONTACT": "violent",
    "BURGLARY-RESIDENCE": "property",
    "BURGLARY-BUSINESS": "property",
    "UUMV": "property",
    "THEFT FROM MOTOR VEHICLE": "property",
    "THEFT OF MOTOR VEHICLE PARTS OR ACCESSORIES": "property",
    "ALL OTHER LARCENY": "property",
    "SHOPLIFTING": "property",
    "THEFT OF BUILDING": "property",
    "POCKET-PICKING": "property",
    "PURSE-SNATCHING": "property",
    "THEFT OF COIN-OPERATED MACHINE OR DEVICE": "property",
}


def _soql_list(values):
    return ",".join("'" + v.replace("'", "''") + "'" for v in values)


def download_incidents(start: str, end: str, timeout: int = 900, page: int = 200_000) -> pd.DataFrame:
    """Offense records with ``start <= date1 < end`` for the offenses above.

    Dates are ISO strings, e.g. ``"2019-01-01"``. Returns one row per
    offense record as delivered by Socrata (not yet de-duplicated).
    """
    # date1 is stored as text ("2026-09-01 00:00:00.0000000"), so bounds must
    # use the same space-separated format to compare correctly.
    where = (
        f"date1 >= '{start} 00:00:00' AND date1 < '{end} 00:00:00'"
        f" AND nibrs_crime IN ({_soql_list(CRIME_TYPES)})"
    )
    select = "servnumid,incidentnum,date1,nibrs_crime,premise,geocoded_column"
    parts, offset = [], 0
    while True:
        res = requests.get(
            SOCRATA_URL,
            params={"$select": select, "$where": where, "$limit": page,
                    "$offset": offset, "$order": "servnumid"},
            timeout=timeout,
        )
        res.raise_for_status()
        rows = res.json()
        parts.append(pd.DataFrame(rows))
        if len(rows) < page:
            break
        offset += page
    return pd.concat(parts, ignore_index=True)


def clean_incidents(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per incident (most serious listed offense), ungeocoded rows dropped."""
    df = raw.copy()
    geo = df["geocoded_column"].apply(lambda d: d if isinstance(d, dict) else {})
    df["lat"] = pd.to_numeric(geo.apply(lambda d: d.get("latitude")), errors="coerce")
    df["lon"] = pd.to_numeric(geo.apply(lambda d: d.get("longitude")), errors="coerce")
    df["date"] = pd.to_datetime(df["date1"].str[:10])
    rank = {k: i for i, k in enumerate(CRIME_TYPES)}
    df["rank"] = df["nibrs_crime"].map(rank)
    df = df.sort_values(["incidentnum", "rank"]).drop_duplicates("incidentnum")
    df["crime_type"] = df["nibrs_crime"].map(CRIME_TYPES)
    df = df[df["lat"].notna() & (df["lat"] != 0)]
    keep = ["incidentnum", "date", "nibrs_crime", "crime_type", "premise", "lat", "lon"]
    return df[keep].sort_values("date").reset_index(drop=True)


def download_pids(timeout: int = 120) -> gpd.GeoDataFrame:
    """All Dallas public improvement district polygons (WGS84)."""
    res = requests.get(
        PID_URL,
        params={"where": "1=1", "outFields": "Name,Date_Exp,Acres", "outSR": 4326, "f": "geojson"},
        timeout=timeout,
    )
    res.raise_for_status()
    return gpd.GeoDataFrame.from_features(res.json()["features"], crs=4326)
