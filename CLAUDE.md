# Bike-Share Demand Forecasting

Forecast hourly pickups and drop-offs at every Citi Bike (NYC) station 1–6 hours ahead, flag stations about to run empty or full, and recommend bike moves. Served through a FastAPI endpoint and a public Streamlit dashboard. 8-week plan, ~10–15 h/week.

Public repo: https://github.com/Ignite-0/Bike-Share-Demand-Forecasting-

## Success criteria

- Beat the "same hour last week" baseline by ≥15% MAE on station-hour pickups, on a held-out final month.
- Empty/full alerts: track precision and recall against real stockouts.
- Public dashboard backed by an API serving fresh predictions.
- Reproducible: one command rebuilds the dataset, another retrains.

## Stack

Python 3.12 (`.venv`, managed with `uv`; not 3.14, since LightGBM/Prophet wheels lag) · DuckDB + Polars/pandas + PyArrow · Parquet (PostgreSQL only once deployed) · LightGBM, scikit-learn, Prophet, SHAP · MLflow · PuLP/OR-Tools (stretch) · FastAPI + Pydantic + Docker · Streamlit + pydeck/Folium · GitHub Actions.

Add dependencies only when a phase needs them. Phase 1 is stdlib-only.

## Commands

```
make data-raw                          # trips (12 months) + weather
make collect                           # one GBFS snapshot
.venv/Scripts/python -m pytest -q tests
uv pip install --python .venv <pkg>    # add a dependency
```

Run modules from the repo root as `python -m src.<pkg>.<module>`.

## Layout

```
data/raw/trips/YYYYMM/*.csv            # monthly trip CSVs (.done marks a finished month)
data/raw/gbfs/status/YYYY-MM-DD/HHMMSS.csv.gz   # live station status snapshots, UTC
data/raw/gbfs/info/YYYY-MM-DD.json     # station metadata, once per day
data/raw/weather/weather_hourly.csv    # Open-Meteo hourly, UTC
data/processed/station_hour.parquet    # Phase 2 output (modeling table)
src/ingest/      download_trips.py, gbfs_collector.py, weather.py
src/features/    build.py (shared by training AND serving)
src/models/      train.py, evaluate.py, predict.py
src/decision/    rebalance.py
src/api/         main.py
app/             streamlit_app.py
notebooks/       01_eda.ipynb, 02_modeling.ipynb
tests/
.github/workflows/  collector.yml (live), ci.yml, retrain.yml (later)
```

`data/` is git-ignored, except `data/raw/gbfs/`, which the collector workflow force-adds and commits to `main` every 10 minutes. **Always `git pull --rebase` before pushing.**

## Data sources

- Trips: `https://s3.amazonaws.com/tripdata/` monthly `YYYYMM-citibike-tripdata.zip`. Window: 2025-10 → 2026-09. Schema is post-2021 (`ride_id`, `rideable_type`, `started_at`, `ended_at`, `start_station_id`, ...).
- Live status: GBFS `https://gbfs.lyft.com/gbfs/1.1/bkn/en/station_status.json` (+ `station_information.json`).
- Weather: Open-Meteo archive API, one point in Midtown (40.7549, -73.9840).
- Holidays: `holidays` package (US/NY).

## Phases

| Week | Phase | Status |
| --- | --- | --- |
| 1 | Data acquisition: trips, GBFS collector, weather | Done (collector live) |
| 2 | Cleaning + aggregation → `station_hour.parquet`, `make data`, data-quality tests | Next |
| 3 | EDA notebook, station clustering (k-means on 24h profile) | |
| 4 | Features in `src/features/build.py` | |
| 4–5 | Baselines → global LightGBM (Poisson/Tweedie) → Prophet on top 20 → quantile LGBM; SHAP | |
| 6 | Decision layer: project stock, flag risk, greedy moves, historical backtest; `recommend(timestamp)` | |
| 7 | FastAPI (`/health`, `/forecast/{station_id}`, `/alerts`, `/recommendations`), Streamlit, Docker, CI, hosting | |
| 8 | Monitoring, weekly retraining with promotion gate, README + write-up | |

Update the status column as phases finish.

## Rules that matter (don't break these)

- **Time-based splits only.** Months 1–9 train, 10–11 validation, 12 test. Rolling-origin CV for tuning. Touch the test month once.
- **Full station × hour grid**, zero-filled, before modeling. Missing zero rows are the most common silent bug.
- **No leakage.** Lags must be ≥ the forecast horizon. Rolling windows are shifted to exclude the target hour. Target encodings are fit on train only.
- **One feature module** (`src/features/build.py`) for both training and serving.
- **Timestamps:** raw data stays in UTC; convert to `America/New_York` in one place (Phase 2) and handle DST explicitly.
- **Cleaning:** drop trips < 60 s or > 24 h, and trips with a missing start or end station. Flag stations active < 30 days as new. Mark GBFS gaps > 30 min as outages; never interpolate silently.
- **Serving weather comes from the forecast, not observations.** Note this mismatch in the write-up.
- **Big data:** use DuckDB or Polars lazy scans over the trip CSVs, never `pd.read_csv` on all of them.
- **Report errors by station cluster and hour**, not just overall MAE. Metrics: MAE, RMSE, WAPE.
- Timebox modeling. The decision layer and deployment matter more.

## Decision layer defaults

Projected stock = current bikes + predicted drop-offs − predicted pickups, clipped to [0, capacity], for each of the next 6 h. Empty risk: ≤ 2 bikes within 3 h. Full risk: ≤ 2 free docks within 3 h. Target range after a move: 30–70% full. Tune thresholds on validation data.
