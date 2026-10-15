"""Download hourly historical weather for central NYC from the Open-Meteo archive API.

Writes data/raw/weather/weather_hourly.csv in UTC; Phase 2 converts to America/New_York so DST is
handled in one place. Rerunning overwrites the file with the full range, so it is safe to repeat.
The archive lags real time by a few days.
Usage: python -m src.ingest.weather --start 2025-10-01 --end 2026-09-30
"""
import argparse
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("data/raw/weather/weather_hourly.csv")
# Midtown Manhattan. One point is enough to start (see project plan, Phase 1).
LAT, LON = 40.7549, -73.9840
VARS = "temperature_2m,apparent_temperature,precipitation,rain,snowfall,wind_speed_10m,weather_code"


def main(start: str, end: str) -> None:
    q = urllib.parse.urlencode({
        "latitude": LAT, "longitude": LON, "start_date": start, "end_date": end,
        "hourly": VARS, "timezone": "GMT", "format": "csv",
        "temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm",
    })
    body = urllib.request.urlopen(f"https://archive-api.open-meteo.com/v1/archive?{q}", timeout=120).read()
    # Open-Meteo's CSV starts with a location metadata block, then a blank line, then the hourly table.
    table = body.decode().split("\n\n", 1)[1]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(table, encoding="utf-8")
    print(f"{OUT}: {table.count(chr(10)) - 1} hourly rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-10-01")
    p.add_argument("--end", default="2026-09-30")
    a = p.parse_args()
    main(a.start, a.end)
