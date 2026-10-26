"""Serving API: forecasts, empty/full alerts, bike moves and the track record.

    .venv/Scripts/python -m uvicorn src.api.main:app --port 8000

Forecasts and the track record are built every 3 h by src.api.publish (GitHub Actions) and published to a
Hugging Face dataset; this process only downloads them (every SYNC_S) and fetches live GBFS status per request
(cached SNAP_TTL), so it fits a small free host. ARTIFACTS=local reads data/serving/ instead (after a local publish).
"""
import json
import os
import shutil
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.decision import rebalance
from src.decision.rebalance import H, covers, live_snapshot, recommend
from src.ingest.gbfs_collector import fetch

ARTIFACTS = os.environ.get("ARTIFACTS", "https://huggingface.co/datasets/himel05/citibike-artifacts/resolve/main/serving")
LOCAL = Path("data/serving")
FILES = ["forecasts.parquet", "scorecard.json", "track.parquet", "p90_sum_correction.json", "alert_thresholds.json"]
SYNC_S = 600
RETRY_S = 30  # after a failed sync (e.g. a cold start hitting a transient HF error)
TIMEOUT_S = 60
SNAP_TTL = 60  # GBFS publishes about once a minute
INFO_TTL = 24 * 3600

state: dict = {"fc": None, "card": None, "track": None, "error": None}
cache: dict = {}
lock = threading.Lock()


def sync() -> None:
    LOCAL.mkdir(parents=True, exist_ok=True)
    if ARTIFACTS != "local":
        for name in FILES:  # download to a temp name, then swap, so readers never see half a file
            with urllib.request.urlopen(f"{ARTIFACTS}/{name}", timeout=TIMEOUT_S) as r,                  open(LOCAL / f"{name}.tmp", "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(LOCAL / f"{name}.tmp", LOCAL / name)
    rebalance.P90_SUM, rebalance.THRESHOLDS = str(LOCAL / FILES[3]), str(LOCAL / FILES[4])
    state["fc"] = pd.read_parquet(LOCAL / "forecasts.parquet")
    state["track"] = pd.read_parquet(LOCAL / "track.parquet")
    state["card"] = json.loads((LOCAL / "scorecard.json").read_text())
    state["error"] = None
    cache.pop("rec_at", None)  # recompute alerts with the new forecasts


def sync_loop() -> None:
    while True:
        try:
            sync()
            time.sleep(SYNC_S)
        except Exception as e:  # keep serving the last good files
            state["error"] = repr(e)
            time.sleep(RETRY_S)


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=sync_loop, daemon=True).start()
    yield


app = FastAPI(title="Citi Bike demand forecasts", lifespan=lifespan)


def need(key: str):
    if state[key] is None:
        raise HTTPException(503, f"loading published forecasts{'; ' + state['error'] if state['error'] else ''}")
    return state[key]


def latest() -> dict:
    """recommend() on the live feed, recomputed at most every SNAP_TTL seconds."""
    fc = need("fc")
    with lock:
        now = time.time()
        if now - cache.get("info_at", 0) > INFO_TTL:
            cache["info"], cache["info_at"] = fetch("station_information"), now
        if now - cache.get("rec_at", 0) > SNAP_TTL:
            snap = live_snapshot(fetch("station_status"), cache["info"])
            t0 = snap["fetched_at"].iloc[0].floor("h")
            if not covers(fc, t0):
                raise HTTPException(503, f"published forecasts don't cover {t0} + {H} h; the 3-hourly build is late")
            cache["rec"], cache["rec_at"] = recommend(snap=snap, fc=fc), now
        return cache["rec"]


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


class Health(BaseModel):
    status: Literal["ok", "loading"]
    forecasts_built_at: str | None
    forecast_window_utc: list[str] | None
    last_snapshot_at: datetime | None
    error: str | None


class Hour(BaseModel):
    ts_utc: datetime
    pu_mean: float
    pu_p90: float
    dr_mean: float
    dr_p90: float


class StationForecast(BaseModel):
    station_id: str
    hours: list[Hour]


class StationRisk(BaseModel):
    station_id: str
    name: str
    lat: float
    lng: float
    bikes: int
    slots: int
    empty: bool
    full: bool
    empty_cautious: bool
    full_cautious: bool
    bikes_6h: float


class Alerts(BaseModel):
    snapshot_at: datetime
    renting_stations: int
    stations: list[StationRisk]


class Move(BaseModel):
    from_id: str
    from_name: str
    to_id: str
    to_name: str
    bikes: int
    km: float
    reason: Literal["empty", "full"]


class Moves(BaseModel):
    snapshot_at: datetime
    moves: list[Move]


class TrackHour(BaseModel):
    ts_utc: datetime
    bikes: int
    proj_1h: float | None
    proj_3h: float | None


@app.get("/health")
def health() -> Health:
    card = state["card"] or {}
    return Health(status="ok" if state["fc"] is not None else "loading", forecasts_built_at=card.get("built_at"),
                  forecast_window_utc=card.get("forecast_window_utc"),
                  last_snapshot_at=cache["rec"]["snapshot_at"] if "rec" in cache else None, error=state["error"])


@app.get("/forecast/{station_id}")
def forecast(station_id: str) -> StationForecast:
    """Next H hours from the current hour: blended mean and P90 pick-ups (pu) and drop-offs (dr)."""
    fc = need("fc")
    t0 = pd.Timestamp.now(tz="UTC").floor("h")
    rows = fc[(fc.station_id == station_id) & fc.ts_utc.between(t0, t0 + pd.Timedelta(hours=H - 1))]
    if rows.empty:
        raise HTTPException(404, f"no forecast for station {station_id} (unknown, or no trip history)")
    return StationForecast(station_id=station_id, hours=records(rows.sort_values("ts_utc").drop(columns=["station_id", "hour"])))


@app.get("/alerts")
def alerts(kind: Literal["empty", "full"] | None = None, cautious: bool = True) -> Alerts:
    """Stations at risk within the look-ahead. cautious=false drops stations flagged only by the P90 path."""
    r = latest()
    a = r["stations"]
    kinds = [kind] if kind else ["empty", "full"]
    cols = kinds + ([f"{k}_cautious" for k in kinds] if cautious else [])
    hit = a[a[cols].any(axis=1)]
    return Alerts(snapshot_at=r["snapshot_at"], renting_stations=len(a),
                  stations=records(hit[list(StationRisk.model_fields)]))


@app.get("/recommendations")
def recommendations() -> Moves:
    r = latest()
    m = r["moves"].rename(columns={"from": "from_id", "to": "to_id"})
    return Moves(snapshot_at=r["snapshot_at"], moves=records(m))


@app.get("/accuracy")
def accuracy() -> dict:
    """Alert precision/recall over the last 7 days against real GBFS stockouts (see src.api.publish)."""
    return need("card")


@app.get("/track/{station_id}")
def station_track(station_id: str) -> list[TrackHour]:
    """Last 48 h: actual bikes at each hour vs the bikes projected for that hour 1 h and 3 h earlier."""
    t = need("track")
    rows = t[t.station_id == station_id]
    if rows.empty:
        raise HTTPException(404, f"no track record for station {station_id}")
    return records(rows.drop(columns="station_id"))
