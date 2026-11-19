# Evaluation protocol: November and December 2026

Written 2026-11-20, before the November trip file is published (expected Dec 3-12). Nothing below is changed
after results are seen; any deviation is reported as a deviation.

## Frozen inputs

- Models `models/*_lightgbm_20261023.txt`, `models/final_config.json` (features, blend weights),
  `models/station_static.parquet`, `models/p90_sum_correction.json`, `models/alert_thresholds.json`
  (levels tuned on GBFS origins Oct 15-25: point 1.5, cautious 0, current stock 1). No retraining or retuning.
- Tier C cutoff rule as in `src/features/build.py` (`PUBLISH_DAY = 15`).

## Forecasts scored

- Target hours: Nov 1 - Nov 30 (scored when the November file appears) and Dec 1 - Dec 31 (when December appears).
- Forecasts rebuilt with `src.models.predict` using archived forecast weather (Open-Meteo historical-forecast API,
  as the served track record does). The model reads the target hour only, so this reproduces what serving issued.
- Check: on every hour covered by `serving/log/` (forecasts as issued), report the max and mean absolute
  difference between issued and rebuilt means. Scheduled runs were dropped often, so the logs cover only part
  of each month; if they disagree materially, score the logged hours only and say so.

## Forecast accuracy

- Metrics: MAE (primary), RMSE, WAPE, bias, P90 exceedance; overall and by cluster, volume tier, day type, hour.
- Baselines: same hour last week; published profile (recent_profile -> profile -> 0); 4-week same-hour mean
  (reference only: needs trips not yet published); the model without the blend.
- Tests: Diebold-Mariano with the Harvey-Leybourne-Newbold correction on daily mean loss differences (h = 1) and
  hourly ones (h = 24); 95% interval of each MAE gain from 4,000 resamples of whole days. Two-sided, alpha 0.05.
- Primary hypothesis H1: the blend beats the published profile on pick-ups, Nov and Dec pooled (daily DM).
  Everything else is secondary and reported without multiplicity correction, labeled as such.
- Thanksgiving (Nov 26-27) stays in; also reported without it.

## Alerts

- Origins: Nov 7 00:00 local (after the Oct 26 - Nov 6 hold-out) through Dec 31, built by
  `src.decision.backtest.pairs` from the committed GBFS snapshots with its existing exclusions (feed gap > 30 min,
  non-renting snapshot, short window).
- Methods at frozen levels: point, cautious (P90 path), current stock. New baseline for onset: the same projection
  with the published profile as the forecast (profile projection), level tuned on Oct 15-25 like the others.
- Metrics: precision, recall, F1 per kind (empty, full), each with a 95% interval from day resamples; onset subset
  = stations above 2 bikes (empty) or 2 free docks (full) at the origin; precision-recall curves over the level
  grid for all methods; secondary: scoring per stockout episode (consecutive zero snapshots) and median lead time.
- H2: at matched precision, point and cautious alerts have higher onset recall than the profile projection.
- Precision is a lower bound: Citi Bike's own rebalancing averts some predicted stockouts.

## Reporting

All results are reported whichever way they go, including H1 or H2 failing.
