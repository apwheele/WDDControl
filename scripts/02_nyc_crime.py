"""Step 2: download NYPD violent and property complaints in Queens.

Writes data/nyc_crime.csv.zip (one row per complaint).
"""

from pathlib import Path

from wddcontrol.nyc import clean_complaints, download_complaints

START, END = "2018-01-01", "2026-07-01"

out = Path(__file__).resolve().parents[1] / "data"
out.mkdir(exist_ok=True)

raw = download_complaints(START, END)
print(f"Raw complaint records: {len(raw):,}")
crime = clean_complaints(raw)
print(f"Unique geocoded complaints: {len(crime):,}")
print(crime.groupby([crime["date"].dt.year, "crime_type"]).size().unstack())
crime.to_csv(out / "nyc_crime.csv.zip", index=False, compression="zip")
