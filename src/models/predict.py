"""Serving forecasts from the frozen Phase 5 models: blended mean and P90 per station x target hour.

    python -m src.models.predict calibrate   # fit the summed-P90 correction on August (val), @aug models
    python -m src.models.predict 2026-10-15 2026-10-29 [weather_csv]   # cache forecasts for a UTC range

Every feature the final model uses depends on the target hour only (tier A + station_id; B-published through
cutoff(T)), never on the forecast origin. One frame of target hours therefore serves every origin; what the
model "knew" is decided by the weather file, which must be a forecast (serving) or an archived forecast (backtest).
"""
import glob
import json
import sys

import duckdb
import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.build import P, TARGETS, TZ, calendar_table, connect, feature_sql
from src.models.evaluate import PREFIX, baseline

CONFIG = "models/final_config.json"
P90_SUM = "models/p90_sum_correction.json"
FORECASTS = f"{P}forecasts.parquet"
WEATHER = "data/raw/weather/weather_historical_forecast.csv"
COLS = "station_id, ts_utc, ts_local, pickups, dropoffs, temp_c, feels_c, precip_mm, snow_cm, wind_kmh, is_holiday"


def load_models(cfg: dict) -> dict:
    """Latest final booster per (target, loss), as saved by final.py run_test."""
    return {(m["target"], m["loss"]): lgb.Booster(model_file=sorted(glob.glob(f"models/{m['target']}_{m['loss']}_lightgbm_*.txt"))[-1])
            for m in cfg["models"].values()}


def features(start_utc: str, end_utc: str, weather_csv: str = WEATHER, batch: int = 500) -> pd.DataFrame:
    """Feature rows for target hours [start, end) UTC through build.feature_sql: station_hour history
    (published trips) + future rows with NULL targets and forecast weather. Stations alive in the last trip month."""
    con = connect()
    t0, t1 = pd.Timestamp(start_utc, tz="UTC"), pd.Timestamp(end_utc, tz="UTC")
    days = pd.date_range(t0.tz_convert(TZ).date(), t1.tz_convert(TZ).date())
    hol = holidays.country_holidays("US", subdiv="NY", years=sorted({d.year for d in days}))
    hol_sql = ", ".join(f"'{d.date()}'" for d in days if d.date() in hol) or "NULL"
    con.sql(f"""CREATE TEMP TABLE fut AS
        WITH w AS (
            SELECT timezone('UTC', strptime(ts, '%Y-%m-%dT%H:%M')) AS ts_utc, temp_c::DOUBLE AS temp_c,
                   feels_c::DOUBLE AS feels_c, precip_mm::DOUBLE AS precip_mm, snow_cm::DOUBLE AS snow_cm,
                   wind_kmh::DOUBLE AS wind_kmh
            FROM read_csv('{weather_csv}', header = true, all_varchar = true,
                          names = ['ts', 'temp_c', 'feels_c', 'precip_mm', 'rain_mm', 'snow_cm', 'wind_kmh', 'code'])
        )
        SELECT s.station_id, g.ts_utc, timezone('{TZ}', g.ts_utc) AS ts_local, NULL::INT AS pickups,
               NULL::INT AS dropoffs, w.temp_c, w.feels_c, w.precip_mm, w.snow_cm, w.wind_kmh,
               coalesce(timezone('{TZ}', g.ts_utc)::DATE IN ({hol_sql}), false) AS is_holiday
        FROM (SELECT station_id FROM '{P}stations.parquet'
              WHERE last_seen >= (SELECT max(last_seen) FROM '{P}stations.parquet') - INTERVAL 30 DAY) s,
             (SELECT unnest(generate_series('{t0}'::TIMESTAMPTZ, '{t1}'::TIMESTAMPTZ - INTERVAL 1 HOUR,
                                            INTERVAL 1 HOUR)) AS ts_utc) g
        LEFT JOIN w USING (ts_utc)""")
    missing = con.sql("SELECT count(*) FROM fut WHERE temp_c IS NULL").fetchone()[0]
    if missing:
        raise ValueError(f"{weather_csv} lacks weather for {missing} future rows; refresh it (src.ingest.weather)")
    d0 = con.sql(f"SELECT min(ts_local)::DATE FROM '{P}station_hour.parquet'").fetchone()[0]
    calendar_table(con, d0, days[-1].date())
    ids = [r[0] for r in con.sql("SELECT DISTINCT station_id FROM fut ORDER BY 1").fetchall()]
    parts = []
    # ponytail: rebuilds tier C from all history every call (~1 min); cache the per-cutoff aggregates if serving needs it faster
    for i in range(0, len(ids), batch):
        con.sql("CREATE OR REPLACE TEMP TABLE batch AS SELECT unnest(?::VARCHAR[]) AS station_id", params=[ids[i:i + batch]])
        src = f"""(SELECT {COLS} FROM '{P}station_hour.parquet' WHERE station_id IN (SELECT station_id FROM batch)
                  UNION ALL SELECT {COLS} FROM fut WHERE station_id IN (SELECT station_id FROM batch))"""
        sql = feature_sql(src, f"'{P}stations.parquet'", f"'{P}station_labels.parquet'")
        parts.append(con.sql(f"SELECT * FROM ({sql}) WHERE ts_utc >= '{t0}'").df())
    df = pd.concat(parts, ignore_index=True)
    df["station_id"] = df["station_id"].astype("category")
    return df


def forecast(df: pd.DataFrame, cfg: dict, models: dict) -> pd.DataFrame:
    """Blended mean (w_h * model + (1 - w_h) * B-published, weights from final_config) and P90 per row."""
    out = df[["station_id", "ts_utc", "hour"]].copy()
    feats = cfg["features"]
    for target in TARGETS:
        t = PREFIX[target]
        w = df["hour"].map({int(h): v for h, v in cfg["blend_weights"][target].items()})
        out[f"{t}_mean"] = w * models[(target, "tweedie1.2")].predict(df[feats]) + (1 - w) * baseline(df, target, "published")
        out[f"{t}_p90"] = models[(target, "q90")].predict(df[feats])
    return out


def forecast_range(start_utc: str, end_utc: str, weather_csv: str = WEATHER) -> pd.DataFrame:
    with open(CONFIG) as f:
        cfg = json.load(f)
    return forecast(features(start_utc, end_utc, weather_csv), cfg, load_models(cfg))


def calibrate() -> None:
    """c_k for k = 1..6 so that the summed bound is exceeded 10% of the time, on August (val) with the Oct-Jul
    (@aug) models: out of sample, and September is never touched."""
    from src.models.data import load

    with open(CONFIG) as f:
        cfg = json.load(f)
    feats, out = cfg["features"], {"fit_on": "Aug 2026 (val) with the @aug models, origins every hour"}
    for target in TARGETS:
        t = PREFIX[target]
        d = load("val", feats + [target, "ts_utc", f"{t}_recent_profile", f"{t}_profile"],
                 where="AND ts_local >= '2026-08-01'").sort_values(["station_id", "ts_utc"], ignore_index=True)
        m = {loss: lgb.Booster(model_file=f"{P}models/{target}_A+sid@aug_{loss}_lightgbm_cpu_tuned.txt")
             for loss in ("tweedie1.2", "q90")}
        w = d["hour"].map({int(h): v for h, v in cfg["blend_weights"][target].items()})
        mean = w * m["tweedie1.2"].predict(d[feats]) + (1 - w) * baseline(d, target, "published")
        hi = np.maximum(m["q90"].predict(d[feats]), mean)
        g = pd.DataFrame({"y": d[target].astype(float), "m": mean, "hi": hi}).groupby(d["station_id"], observed=True)
        ts = d.groupby("station_id", observed=True)["ts_utc"]
        out[target], out[f"{target}_raw_sum_above"] = {}, {}
        for k in range(1, 7):
            s = g.rolling(k).sum().reset_index(level=0, drop=True).sort_index()
            ok = (d["ts_utc"] - ts.shift(k - 1)) == pd.Timedelta(hours=k - 1)  # k contiguous hours, same station
            s = s[ok]
            need = np.where(s.y <= s.m, -np.inf, (s.y - s.m) / (s.hi - s.m).where(s.hi > s.m, 0))
            out[target][k] = float(np.quantile(need, 0.9, method="higher"))
            out[f"{target}_raw_sum_above"][k] = float((s.y > s.hi).mean())
        print(target, {k: round(v, 3) for k, v in out[target].items()},
              "| raw summed P90 exceeded:", {k: round(v, 3) for k, v in out[f"{target}_raw_sum_above"].items()})
    with open(P90_SUM, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "calibrate":
        calibrate()
    else:
        fc = forecast_range(*sys.argv[1:4])
        fc.to_parquet(FORECASTS)
        print(f"{FORECASTS}: {len(fc):,} rows, {fc.station_id.nunique()} stations\n", fc.describe().T.to_string())
