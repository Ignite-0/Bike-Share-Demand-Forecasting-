"""Decision layer: project each station's stock 6 h ahead, flag empty/full risk, recommend greedy bike moves.

    python -m src.decision.rebalance ["2026-10-26 15:50"]   # UTC; latest snapshot at or before it (default: newest)

Projected stock = current bikes + drop-offs - pick-ups, clipped to [0, slots] hour by hour, where slots = bikes +
free docks (disabled bikes/docks can't be used). Empty risk: <= 2 bikes within LOOKAHEAD_H; full risk: <= 2 free
docks (defaults; tuned levels in models/alert_thresholds.json). Cautious variant: cumulative P90 pick-ups (empty) or drop-offs (full), corrected so summed
hourly quantiles keep 10% exceedance (models/p90_sum_correction.json). The current stock counts as hour 0.
Stations with is_renting = 0 or is_installed = 0 are excluded. Thresholds: tuned in src.decision.backtest.
"""
import glob
import json
import sys

import duckdb
import numpy as np
import pandas as pd

from src.features.build import P
from src.ingest.station_hour import norm_id
from src.models.predict import FORECASTS, P90_SUM, WEATHER, forecast_range

H = 6
LOOKAHEAD_H = 3
# alert levels: bikes (empty) / free docks (full); defaults until src.decision.backtest writes THRESHOLDS
DEFAULT_LEVELS = {"empty": 2, "full": 2, "empty_cautious": 2, "full_cautious": 2}
THRESHOLDS = "models/alert_thresholds.json"
BAND = (0.3, 0.7)  # target fill range after a move
MAX_KM = 2.0  # donor/receiver search radius
TRUCK = 20  # max bikes per move
MIN_MOVE = 3  # smaller moves aren't worth a truck stop
STATUS = f"{P}gbfs_status.parquet"


def info_sql() -> str:
    """GBFS station_id -> trip station_id (via norm_id(short_name)), name, position; latest metadata file."""
    info = sorted(glob.glob("data/raw/gbfs/info/*.json"))[-1]
    return (f"SELECT station_id AS gbfs_id, {norm_id('short_name')} AS station_id, name, lat, lon AS lng "
            f"FROM read_json('{info}')")


def snapshot(ts=None, status: str = STATUS) -> pd.DataFrame:
    """Latest snapshot at or before ts (UTC), renting stations only, keyed by trip station_id."""
    ts = None if ts is None else pd.Timestamp(ts)
    ts = ts if ts is None or ts.tz else ts.tz_localize("UTC")  # naive = UTC
    when = f"'{ts.tz_convert('UTC')}'::TIMESTAMPTZ" if ts is not None else "now()"
    return duckdb.sql(f"""
        WITH snap AS (SELECT max(fetched_at) AS f FROM '{status}' WHERE fetched_at <= {when}),
             info AS ({info_sql()})
        SELECT i.station_id, i.name, i.lat, i.lng, s.fetched_at, s.num_bikes_available AS bikes,
               s.num_bikes_available + s.num_docks_available AS slots
        FROM '{status}' s JOIN snap ON s.fetched_at = snap.f JOIN info i ON i.gbfs_id = s.station_id
        WHERE s.is_renting = 1 AND s.is_installed = 1
          AND s.num_bikes_available + s.num_docks_available > 0""").df()


def project(stock0, net_cum, slots) -> np.ndarray:
    """Stock after each hour, clipped to [0, slots] hour by hour. net_cum: (n, H) cumulative drop-offs - pick-ups."""
    stock0, slots = np.asarray(stock0, float), np.asarray(slots, float)
    step = np.diff(np.asarray(net_cum, float), axis=1, prepend=0)
    out, s = np.empty_like(step), stock0
    for h in range(step.shape[1]):
        s = np.clip(s + step[:, h], 0, slots)
        out[:, h] = s
    return out


def assess(snap: pd.DataFrame, fc: pd.DataFrame, corr: dict, lookahead: int = LOOKAHEAD_H) -> pd.DataFrame:
    """Per row of snap (station_id, t0 = first target hour, bikes, slots [, frac]): min/max projected bikes over hours 0..lookahead for the
    point path, the cautious empty path (P90 pick-ups) and the cautious full path (P90 drop-offs).
    Stations with no forecast (no trip history yet) get zero flow, so only their current stock can flag them."""
    n, hs = len(snap), np.arange(1, H + 1)
    grid = snap[["station_id", "t0"]].loc[snap.index.repeat(H)].reset_index(drop=True)
    grid["h"] = np.tile(hs, n)
    grid["ts_utc"] = grid["t0"] + pd.to_timedelta(grid["h"] - 1, unit="h")
    cols = ["pu_mean", "pu_p90", "dr_mean", "dr_p90"]
    g = grid.merge(fc[["station_id", "ts_utc"] + cols], on=["station_id", "ts_utc"], how="left")[cols].fillna(0)
    a = {c: g[c].to_numpy(float, copy=True).reshape(n, H) for c in cols}  # grid order = snap order x h
    frac = snap["frac"].to_numpy() if "frac" in snap else np.ones(n)
    for c in cols:  # the first hour is partly over at a mid-hour snapshot
        a[c][:, 0] *= frac
    cum = {}
    for t in ("pu", "dr"):
        c = np.array([corr["pickups" if t == "pu" else "dropoffs"][str(h)] for h in hs])
        cum[t] = a[f"{t}_mean"].cumsum(1)
        cum[f"{t}90"] = cum[t] + c * (np.maximum(a[f"{t}_p90"], a[f"{t}_mean"]).cumsum(1) - cum[t])
    b0, slots = snap["bikes"].to_numpy(float), snap["slots"].to_numpy(float)
    paths = {"point": project(b0, cum["dr"] - cum["pu"], slots),
             "empty90": project(b0, cum["dr"] - cum["pu90"], slots),
             "full90": project(b0, cum["dr90"] - cum["pu"], slots)}
    out = snap.copy()
    for name, p in paths.items():
        win = np.column_stack([b0, p[:, :lookahead]])
        out[f"min_{name}"], out[f"max_{name}"] = win.min(1), win.max(1)
    out["bikes_6h"] = paths["point"][:, -1]
    return out


def risk_scores(a: pd.DataFrame) -> pd.DataFrame:
    """Lowest projected bikes (empty) or free docks (full) within the look-ahead; alert when <= the level.
    persistence_* use the current stock only (the baseline in the backtest)."""
    return pd.DataFrame({"empty": a["min_point"], "full": a["slots"] - a["max_point"],
                         "empty_cautious": a["min_empty90"], "full_cautious": a["slots"] - a["max_full90"],
                         "empty_persistence": a["bikes"], "full_persistence": a["slots"] - a["bikes"]})


def levels() -> dict:
    try:
        with open(THRESHOLDS) as f:
            return json.load(f)["levels"]
    except FileNotFoundError:
        return DEFAULT_LEVELS


def flag(a: pd.DataFrame, lv: dict | None = None) -> pd.DataFrame:
    lv, sc = lv or levels(), risk_scores(a)
    return a.assign(**{k: sc[k] <= lv[k] for k in DEFAULT_LEVELS})


def moves(a: pd.DataFrame, max_km: float = MAX_KM, truck: int = TRUCK, min_move: int = MIN_MOVE) -> pd.DataFrame:
    """Greedy: each at-risk station, most urgent first, is paired with the nearest station that can absorb the move
    without leaving the 30-70% band at any projected hour (point path). Shifting stock by q shifts the whole path.
    Each station takes part in at most one move."""
    # ponytail: one move per at-risk station, straight-line distance, no truck routing; OR-Tools VRP is the upgrade
    a = a.reset_index(drop=True)
    lo, hi = np.ceil(BAND[0] * a["slots"]), np.floor(BAND[1] * a["slots"])
    spare = (a["min_point"] - lo).clip(lower=0).to_numpy(float, copy=True)  # bikes a station can give and stay >= 30%
    room = (hi - a["max_point"]).clip(lower=0).to_numpy(float, copy=True)  # bikes it can take and stay <= 70%
    need = np.where(a["empty"], lo - a["min_point"], 0)
    excess = np.where(a["full"] & ~a["empty"], a["max_point"] - hi, 0)
    xy = np.column_stack([a["lat"] * 111.32, a["lng"] * 111.32 * np.cos(np.radians(40.73))])
    urgency = np.where(need > 0, a["min_point"] / a["slots"], np.where(excess > 0, 1 - a["max_point"] / a["slots"], np.inf))
    rows = []
    for i in np.argsort(urgency, kind="stable"):
        if need[i] <= 0 and excess[i] <= 0:
            break
        receiving = need[i] > 0
        want = min(need[i], room[i]) if receiving else min(excess[i], spare[i])
        pool = spare if receiving else room
        d = np.hypot(*(xy - xy[i]).T)
        ok = (pool > 0) & (d <= max_km) & (need <= 0) & (excess <= 0)
        if want < min_move or not ok.any():
            continue
        j = np.flatnonzero(ok)[np.argmin(d[ok])]
        q = int(min(want, pool[j], truck))
        if q < min_move:
            continue
        src, dst = (j, i) if receiving else (i, j)
        spare[[i, j]] = room[[i, j]] = 0
        need[i] = excess[i] = 0
        rows.append({"from": a.at[src, "station_id"], "from_name": a.at[src, "name"],
                     "to": a.at[dst, "station_id"], "to_name": a.at[dst, "name"], "bikes": q, "km": round(d[j], 2),
                     "reason": "empty" if receiving else "full"})
    return pd.DataFrame(rows, columns=["from", "from_name", "to", "to_name", "bikes", "km", "reason"])


def recommend(timestamp=None, weather_csv: str = WEATHER, lv: dict | None = None) -> dict:
    """Entry point for the API: the latest snapshot at or before timestamp (UTC) -> station risks and moves."""
    snap = snapshot(timestamp)
    at = snap["fetched_at"].iloc[0].tz_convert("UTC")
    snap["t0"], snap["frac"] = at.floor("h"), 1 - (at - at.floor("h")) / pd.Timedelta(hours=1)
    try:  # reuse the cached range if it covers the six hours
        fc = pd.read_parquet(FORECASTS)
        if not ((fc.ts_utc.min() <= at.floor("h")) and (fc.ts_utc.max() >= at.floor("h") + pd.Timedelta(hours=H - 1))):
            raise FileNotFoundError
    except FileNotFoundError:
        fc = forecast_range(str(at.floor("h")), str(at.floor("h") + pd.Timedelta(hours=H)), weather_csv)
    with open(P90_SUM) as f:
        corr = json.load(f)
    a = flag(assess(snap, fc, corr), lv)
    return {"snapshot_at": at, "stations": a, "moves": moves(a)}


if __name__ == "__main__":
    r = recommend(sys.argv[1] if len(sys.argv) > 1 else None)
    a = r["stations"]
    print(f"snapshot {r['snapshot_at']}: {len(a)} renting stations | alerts: "
          + ", ".join(f"{c} {int(a[c].sum())}" for c in ("empty", "full", "empty_cautious", "full_cautious")))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(r["moves"].head(20).to_string(index=False), f"\n{len(r['moves'])} moves, {r['moves'].bikes.sum()} bikes")
