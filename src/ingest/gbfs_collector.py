"""Snapshot Citi Bike live station status (GBFS). Run every 10 minutes (see .github/workflows/collector.yml).

Writes one gzipped CSV per run: data/raw/gbfs/status/YYYY-MM-DD/HHMMSS.csv.gz (UTC), every row
stamped with fetched_at. One file per snapshot keeps git commits small and appends crash-safe.
Station metadata is saved once per day to data/raw/gbfs/info/YYYY-MM-DD.json.
Usage: python -m src.ingest.gbfs_collector
"""
import csv
import gzip
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FEED = "https://gbfs.lyft.com/gbfs/1.1/bkn/en/"
OUT = Path("data/raw/gbfs")
COLS = [
    "station_id", "num_bikes_available", "num_ebikes_available", "num_bikes_disabled",
    "num_docks_available", "num_docks_disabled", "is_installed", "is_renting", "is_returning",
    "last_reported",
]


def fetch(name: str) -> dict:
    with urllib.request.urlopen(FEED + name + ".json", timeout=30) as r:
        return json.load(r)


def main() -> None:
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y-%m-%d")

    status = fetch("station_status")
    path = OUT / "status" / day / (now.strftime("%H%M%S") + ".csv.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    fetched_at = now.isoformat(timespec="seconds")
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fetched_at", *COLS])
        for s in status["data"]["stations"]:
            w.writerow([fetched_at, *(s.get(c) for c in COLS)])
    print(f"{path}: {len(status['data']['stations'])} stations")

    info = OUT / "info" / f"{day}.json"
    if not info.exists():
        info.parent.mkdir(parents=True, exist_ok=True)
        info.write_text(json.dumps(fetch("station_information")["data"]["stations"]))
        print(f"{info}: saved")


if __name__ == "__main__":
    main()
