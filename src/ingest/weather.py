"""Download hourly historical weather for central NYC from the Open-Meteo archive API.

Writes data/raw/weather/weather_hourly.csv in UTC; Phase 2 converts to America/New_York so DST is
handled in one place. Rerunning overwrites the file with the full range, so it is safe to repeat.
The archive lags real time by a few days.
Usage: python -m src.ingest.weather --start 2025-10-01 --end 2026-11-01

--api historical-forecast: archived forecasts (what serving would have seen), for the Phase 6 backtest.
--api forecast: the live forecast (past 2 days + today + next 2), for serving. Both write weather_<api>.csv.
"""
import argparse
import time
from datetime import datetime, timedelta, timezone
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("data/raw/weather/weather_hourly.csv")
APIS = {"archive": "https://archive-api.open-meteo.com/v1/archive",
        "historical-forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
        "forecast": "https://api.open-meteo.com/v1/forecast"}
# Midtown Manhattan. One point is enough to start (see project plan, Phase 1).
LAT, LON = 40.7549, -73.9840
VARS = "temperature_2m,apparent_temperature,precipitation,rain,snowfall,wind_speed_10m,weather_code"


def _get(url: str, tries: int = 3) -> bytes:
    # Open-Meteo has outages (TLS handshake timeouts); retry before giving up.
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except OSError as e:  # URLError and TimeoutError are both OSError
            if attempt == tries - 1:
                raise
            print(f"{url.split('?')[0]}: {e}; retrying in {30 * (attempt + 1)} s")
            time.sleep(30 * (attempt + 1))


def main(start: str, end: str, api: str = "archive", out: Path | None = None) -> Path:
    units = {"hourly": VARS, "timezone": "GMT", "format": "csv",
             "temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm"}
    loc = {"latitude": LAT, "longitude": LON}
    if api == "forecast":
        try:
            body = _get(f"{APIS[api]}?{urllib.parse.urlencode({**loc, 'past_days': 2, 'forecast_days': 3, **units})}")
        except OSError:
            # api.open-meteo.com was down for a day in Nov 2026. The historical-forecast endpoint also serves the
            # upcoming days (same models), so ask it for the same 5-day window.
            today = datetime.now(timezone.utc).date()
            span = {"start_date": today - timedelta(days=2), "end_date": today + timedelta(days=2)}
            print("forecast API down; using historical-forecast for the same window")
            body = _get(f"{APIS['historical-forecast']}?{urllib.parse.urlencode({**loc, **span, **units})}")
    else:
        body = _get(f"{APIS[api]}?{urllib.parse.urlencode({**loc, 'start_date': start, 'end_date': end, **units})}")
    # Open-Meteo's CSV starts with a location metadata block, then a blank line, then the hourly table.
    table = body.decode().split("\n\n", 1)[1]
    out = out or (OUT if api == "archive" else OUT.with_name(f"weather_{api.replace('-', '_')}.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table, encoding="utf-8")
    print(f"{out}: {table.count(chr(10)) - 1} hourly rows")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-10-01")
    p.add_argument("--end", default="2026-11-01")
    p.add_argument("--api", default="archive", choices=list(APIS))
    a = p.parse_args()
    main(a.start, a.end, a.api)
