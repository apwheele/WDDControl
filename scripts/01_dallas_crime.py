"""Step 1: download Dallas PD violent and property incidents and PID boundaries.

Writes data/dallas_crime.csv.zip (one row per incident) and data/dallas_pids.gpkg.
"""

from pathlib import Path

from wddcontrol.dallas import clean_incidents, download_incidents, download_pids

START, END = "2019-01-01", "2026-10-01"

out = Path(__file__).resolve().parents[1] / "data"
out.mkdir(exist_ok=True)

pids = download_pids()
pids.to_file(out / "dallas_pids.gpkg", driver="GPKG")
print(f"PIDs: {len(pids)}")

raw = download_incidents(START, END)
print(f"Raw offense records: {len(raw):,}")
crime = clean_incidents(raw)
print(f"Unique geocoded incidents: {len(crime):,}")
print(crime.groupby([crime["date"].dt.year, "crime_type"]).size().unstack())
crime.to_csv(out / "dallas_crime.csv.zip", index=False, compression="zip")
