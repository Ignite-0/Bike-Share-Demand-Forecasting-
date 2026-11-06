# October 2026: first out-of-sample month

October trips (published early November) are the first month the frozen Phase 5 model never saw. They land in a
new `live` split (`LIVE_FROM = 2026-10-01` in `src/features/build.py`); train/val/test are unchanged, and September
was not touched again.

Full table: `reports/october_results.csv` (overall, cluster, volume tier, day type, hour; both targets).
Reproduce: `python -m src.models.evaluate --live data/processed/forecasts_oct15_28.parquet`.

## What was scored

- Forecasts: the serving blend (mean) and P90, cached for Oct 15-28 (UTC) **before** October trips existed, with
  archived-forecast weather (Open-Meteo historical-forecast API). This is the file the Phase 6 backtest used, kept
  as `data/processed/forecasts_oct15_28.parquet`.
- Actuals: rebuilt `station_hour`. 788,924 station-hours at 2,358 stations. 27,572 October rows (3.4%) have no
  forecast: stations not alive in the 30 days before September's end (new or reopened in October).
- Baselines: "same hour last week" (`lag168` -> `same_hour_4w`) and the published profile (`recent_profile` ->
  `profile`), exactly as in Phase 5.

## Results

| | MAE | RMSE | WAPE | bias | vs last week | vs profile | P90 above |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pick-ups | 1.253 | 2.439 | 0.443 | -2.0% | **20.5%** | 5.3% | 11.8% |
| Drop-offs | 1.254 | 2.435 | 0.443 | -1.7% | **20.3%** | 5.1% | 12.1% |

September (test) for comparison: 24.0% / 23.4% better than last week, 13% / 8.5% better than the profile,
P90 11.5% above. October clears the 15% target (H1) but the gap to the profile shrank.

By cluster (pick-ups; drop-offs within 1 point):

| Cluster | MAE | WAPE | vs last week | vs profile | P90 above |
| --- | --- | --- | --- | --- | --- |
| 0 residential | 1.262 | 0.445 | 22.2% | 6.0% | 11.3% |
| 1 evening-heavy | 0.946 | 0.526 | 21.5% | 7.3% | 12.4% |
| 2 balanced | 1.596 | 0.399 | 22.1% | 5.6% | 12.0% |
| 3 business | 1.675 | 0.418 | 14.9% | 2.6% | 10.9% |
| -1 low volume | 0.204 | 1.667 | -4.7% | 3.5% | 11.9% |
| new (opened after June) | 0.695 | 0.662 | -8.7% | -13.3% | 10.9% |

By hour (pick-ups): the gain over last week is 19-27% from 10:00 to 23:00, 14-17% in the 7-9 h peak and
0-8% from 4:00 to 6:00, where volumes are tiny. Against the profile the model adds most in the evening and at
night (7-13% at 19-23 h) and nothing at 5-6 h (-1%). Drop-offs follow the same shape.

## Reading it

- **Smaller edge over the profile than in September.** For target hours from Oct 16 the recent profile is built
  from September trips, which already show the post-summer return-to-work level. In September the profile was
  still averaging August, so the model's seasonal correction was worth more. Low bias (-2%) shows the ~9%
  under-forecast seen after the September change is gone.
- **New stations are the weak spot.** Stations opened after June lose to both baselines: the model has no
  `station_id` history for them until the next retrain. Business stations gain least.
- **P90 runs slightly hot**: 11.8-12.1% above versus a 10% target, a bit worse than September (11.5%). The
  summed-P90 correction used by the alerts is fit separately and is not affected.
- Day type: weekends gain 29% over last week, weekdays 17%. No holiday falls inside Oct 15-28.

## Caveats

- Forecast weather, not observations, as in serving. Two weeks, overlapping a single season: indicative.
- Rebuilding `stations.parquet` with the newest GBFS metadata changed capacity for some stations (17 renting
  stations now report 0, e.g. 6659.01). Capacity is a model feature, so regenerated forecasts for the same past
  hours differ (about 2% of rows, up to 22 trips at one station). That is why the pre-October cache is scored
  here, not a regenerated one.
  Serving now reads capacity/lat/lng from `models/station_static.parquet` (the station table uploaded with the frozen
  models on Oct 26; 73 stations differ from today's, 14 dropped to 0), falling back to `stations.parquet` only for
  stations opened since. With it, regenerated P90 for past hours matches the cache exactly.
- The blended mean can still move slightly on a rebuild: each station's grid ends at its last trip, so the last day
  of the newest month lacked late-evening zero hours (Sep 30 had them only after October arrived). The recent
  profile shifted by about 1/19 at 19-23 h for ~4% of rows. Logged forecasts (`serving/log/`) avoid both effects.

## Alert backtest rerun (`reports/backtest_alerts.csv`)

`gbfs_status` was rebuilt from origin/main's snapshots (Oct 15 - Nov 6 13:00 UTC) and forecasts were regenerated
for Oct 15 - Nov 7 with archived-forecast weather. The tune set (origins before Oct 26) gave the same levels as
before (point 1.5, cautious 0); they were not changed. The hold-out grew from 6 to 201 origins (Oct 26 - Nov 6):

| | precision | recall | F1 | persistence (tuned) P / R / F1 |
| --- | --- | --- | --- | --- |
| Empty, point 1.5 | 0.704 | 0.729 | 0.716 | 0.732 / 0.635 / 0.680 |
| Empty, cautious 0 | 0.587 | 0.858 | 0.697 | |
| Full, point 1.5 | 0.780 | 0.764 | 0.772 | 0.790 / 0.681 / 0.731 |
| Full, cautious 0 | 0.717 | 0.870 | 0.786 | |

At stations not at risk now (which persistence can never flag), point alerts catch 20% of empty and full
stockouts at 59% / 73% precision. Precision is a lower bound because Citi Bike rebalances too.
