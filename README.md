# Citi Bike demand forecasting and stockout alerts

Forecasts hourly pick-ups and drop-offs at every Citi Bike station in New York 1–6 hours ahead, flags stations
about to run empty or full, and suggests bike moves to prevent it.

**Live:** [dashboard](https://citibik.streamlit.app/) · [API docs](https://citibike-api-vk26.onrender.com/docs)
(the first request after a quiet spell takes about a minute while the API wakes up)

| Result | Value |
|---|---|
| Forecast accuracy, test month (Sep 2026, scored once) | **24% lower error** than "same hour last week" on pick-ups (target ≥ 15%); 23% on drop-offs |
| Forecast accuracy, October 2026 (never seen, made with forecast weather) | **20.5% lower error** on pick-ups, 20.3% on drop-offs |
| Against the station's published average (the best baseline production could use) | 13% lower error in September, 5% in October |
| Empty-station alerts, Oct 26 – Nov 6 (out of sample, 201 hourly checks) | 73% of stockouts warned in advance vs 64% for "nearly empty now"; 70% of alerts came true |
| Full-station alerts, same period | 76% warned vs 68%; 78% of alerts came true |
| Stockouts caught before the station was low at all | ~20% (the "nearly empty now" rule can't catch any) |

The dashboard's **Forecast accuracy** tab shows each month the model never saw, by hour and station type.

## How it works

```mermaid
flowchart LR
    T[Monthly trip files<br/>50M trips] --> SH[station_hour<br/>21.4M rows]
    W[Weather forecast<br/>Open-Meteo] --> F
    SH --> F[Features<br/>src/features/build.py]
    F --> M[LightGBM + profile blend<br/>mean and P90]
    G[Live station status<br/>GBFS, every 10 min] --> D
    M --> D[Decision layer<br/>project stock, alerts, moves]
    D --> API[FastAPI on Render] --> UI[Streamlit dashboard]
    G --> TR[Track record<br/>alerts vs real stockouts] --> API
```

1. **Data.** Thirteen months of trips (Oct 2025 – Oct 2026, 50.3M after cleaning) are aggregated into a full
   station × hour grid with zeros, weather and holidays. October arrived after the model was frozen and is
   kept apart as an out-of-sample month. A GitHub Actions job has saved live station status
   every 10 minutes since Oct 15, 2026; that is the ground truth for stockouts.
2. **Forecast.** One LightGBM model per target on calendar, weather, station location/capacity and station ID,
   with a Tweedie loss, blended hour by hour with each station's average profile from published trip months.
   Quantile models give a cautious 90th-percentile bound.
3. **Decision layer.** Projected stock = bikes now + forecast drop-offs − forecast pick-ups, clipped to the
   station's free slots hour by hour. A station is flagged if it is projected to hit the alert level within
   3 hours; the cautious variant uses the 90th-percentile demand. Moves pair each at-risk station with the
   nearest station within 2 km that stays 30–70% full.
4. **Serving.** Every 3 hours a GitHub Actions job builds the next 48 hours of forecasts and the 7-day track
   record and publishes them to a Hugging Face dataset. Each run also keeps the next 6 hours of forecasts as
   issued, so later months are scored on forecasts frozen in advance. The API downloads those files and reads live station
   status on each request (cached for a minute), so it runs on a free host.

## What the data showed

![Hourly profile](reports/figures/01_hourly_profile.png)

- **Commute vs leisure.** Weekdays peak at 8:00 and 17:00; weekends have one plateau from noon to 16:00.
  Weekday holidays sit halfway between, so holidays get their own flag.
- **Season and weather.** June had 4.4× the trips of February. Daily trips track temperature (r = 0.84).
  Light rain cuts an hour's trips to 58% of a dry hour, heavy rain to 37%.
- **Mirror flows.** Stations that drain in the morning fill in the evening (correlation −0.94), which is
  what rebalancing fights.
- **Stockouts.** At any moment about 10% of stations are full and 4% empty, so full matters as much as empty.

![Rain effect](reports/figures/04_rain_effect.png)

## Modeling decisions

**What does production actually know?** Citi Bike publishes trips once a month, 3–12 days after the month ends.
A live forecast can't use last hour's or last week's trips. Features were split by availability:

| Tier | Contents | Used |
|---|---|---|
| A | Calendar, holiday, daylight, weather, station location and capacity | In the model |
| C | Station profiles from trip months already published when the forecast is made | Only as the blend |
| B | Fresh trip lags (6 h to 2 weeks) | Upper bound only, never reported as production accuracy |

Findings that shaped the final model (full log in [`reports/experiments.csv`](reports/experiments.csv)):

- **Trip-history features made forecasts worse.** Every model reading past trip levels over-predicted summer
  by 8–26%: the training months were all spring growth, where demand outran its recent level by 1.3–1.8×, and
  the trees carried that growth into a flat summer. The published profile helps only as a linear blend.
  A side effect: serving needs no live trip data.
- **Rain is where the model earns its keep:** in heavy rain it is 38–40% better than the profile. Its weak
  spot is the 7–9 a.m. peak, where the blend leans on the profile.
- **Alternatives that lost:** CatBoost (6–9% worse, much slower even on GPU), squared-error and absolute
  losses, and Prophet (22% worse on the 20 busiest stations).
- **The 90th-percentile bound is calibrated:** 11.5% of September hours were above it (target 10%). Summed
  over several hours it was too wide, so the decision layer shrinks it (fit on August, out of sample).
- **The test month was used once**, after the configuration was frozen, and is guarded in code.

Test-month detail ([`reports/test_results.csv`](reports/test_results.csv)): pick-ups MAE 1.37 vs 1.80 for
"same hour last week" and 1.58 for the published profile. Without the Sep 26–28 storm (51 mm of rain on Sep 27)
the gain is 22.1% on pick-ups and 21.7% on drop-offs.

![SHAP summary](reports/figures/shap_beeswarm_pickups.png)

## Alerts and the track record

Alert levels were tuned by F1 on Oct 15–25 snapshots ([`reports/backtest_alerts.csv`](reports/backtest_alerts.csv))
and frozen. The dashboard's **Track record** tab rescores the last 7 days every 3 hours, counting only alerts
made after Oct 26 so the levels are scored out of sample: an alert counts as correct if the station really hit
0 bikes (or 0 free docks) in a saved snapshot within 3 hours. The comparison is "alert if nearly empty/full
now", which is what you'd do without a forecast.

Scored on Oct 26 – Nov 6, after the levels were frozen (201 hourly checks, out of sample):

| Oct 26 – Nov 6 | Alerts that came true | Stockouts warned | Caught early |
|---|---|---|---|
| Empty: forecast | 70% | 73% | 21% |
| Empty: forecast, cautious (P90) | 59% | 86% | 64% |
| Empty: "nearly empty now" | 73% | 64% | 0% |
| Full: forecast | 78% | 76% | 20% |
| Full: forecast, cautious (P90) | 72% | 87% | 63% |
| Full: "nearly full now" | 79% | 68% | 0% |

The forecast trades 1–3 points of precision for 8–9 points of recall, and it is the only one that warns about
stations that are still healthy. The cautious variant catches most stockouts at the cost of more false alarms.

## Limitations

- **Serving weather comes from the forecast; training used observed weather.** October, scored with forecast
  weather, came in at 20.5% better than last week against 24.0% in September (below).
- **Precision is a lower bound.** Citi Bike's own trucks and Bike Angels move bikes after an alert, so some
  "false alarms" were stockouts that someone prevented.
- **Seasonal turns lag.** September ran ~9% above forecast as people returned to work; by October the bias was
  back to −2%. Monthly retraining is the fix.
- **Not in the model:** planned events (Summer Streets caused the largest single errors, up to 4× normal
  demand), station closures, and new stations until the next retrain (they get zero flow; ~5% of stations).
- **One weather point** (Midtown) for the whole city.
- **Moves** are one per at-risk station, straight-line distance, no truck routing.
- **The 7-day track record is rebuilt** with archived weather forecasts. Station capacity is frozen at the
  values the model was trained with, so a data refresh can't change past forecasts; from November, forecasts
  are also logged as issued.
- **The 90th-percentile bound runs a little loose live:** 11.8–12.1% of October hours were above it (target 10%).

## October accuracy

October 2026 is the first month the frozen model never saw. Forecasts for Oct 15–28 were made before its trips
were published, with archived forecast weather, and scored against the real counts (789k station-hours):

| October | MAE | vs "same hour last week" | vs published average | Above P90 |
|---|---|---|---|---|
| Pick-ups | 1.25 | **20.5% lower** | 5.3% lower | 11.8% |
| Drop-offs | 1.25 | **20.3% lower** | 5.1% lower | 12.1% |

- The 15% target holds out of sample. The edge over the published average shrank (13% in September) because the
  October average already includes September's back-to-work level.
- Residential, evening-heavy and balanced stations gain 21–22% over last week; business stations 15%.
- Stations opened after the training months lose to both baselines until the next retrain.
- Breakdowns by hour and station type are in the dashboard's **Forecast accuracy** tab
  ([`reports/october_results.csv`](reports/october_results.csv)).

## Reproduce

Python 3.12 with [uv](https://docs.astral.sh/uv/). Data stays out of git; the commands rebuild it.

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
python -m src.ingest.download_trips --months 13          # 9.3 GB of trip CSVs
python -m src.ingest.weather --start 2025-10-01 --end 2026-11-01
python -m src.ingest.station_hour                        # cleaned trips + station_hour (~1.5 min)
python -m src.features.build                             # feature table (~70 s, 4.8 GB peak)
python -m src.models.final prepare                       # tree counts from the val months (frozen config)
python -m src.models.final test                          # retrain Oct-Aug, the single test run (refuses a rerun)
python -m src.models.evaluate --live <cached forecasts>  # score forecasts on months after the freeze
python -m src.api.publish                                # forecasts + track record -> data/serving
ARTIFACTS=local python -m uvicorn src.api.main:app       # API on :8000
python -m streamlit run app/streamlit_app.py             # dashboard on :8501
python -m pytest -q tests
```

Training extras (SHAP, Prophet, matplotlib) aren't in `requirements.txt`, which covers the API and tests.

## Layout

```
src/ingest/      trip download, GBFS collector, weather, station_hour build
src/features/    build.py: the one feature module for training and serving
src/models/      training, evaluation, tuning, SHAP, Prophet, final run, serving forecasts
src/decision/    stock projection, alerts, moves, alert backtest
src/api/         FastAPI app and the publish job
app/             Streamlit dashboard
notebooks/       01_eda.ipynb
reports/         figures, experiment log, test results, alert backtest, October accuracy
.github/         collector (every 10 min), serve (every 3 h), CI
```

## Data sources

| Data | Source | Terms |
|---|---|---|
| Trips, Oct 2025 – Oct 2026 | [Citi Bike System Data](https://citibikenyc.com/system-data), monthly files from [s3.amazonaws.com/tripdata](https://s3.amazonaws.com/tripdata/index.html) | [Citi Bike Data License Agreement](https://citibikenyc.com/data-sharing-policy) |
| Live station status and station info, every 10 min since Oct 15, 2026 | Citi Bike's public [GBFS feed](https://gbfs.lyft.com/gbfs/1.1/bkn/en/station_status.json) (format: [GBFS](https://github.com/MobilityData/gbfs)) | Citi Bike Data License Agreement |
| Hourly weather: observed (training), archived forecasts (track record), live forecast (serving) | [Open-Meteo](https://open-meteo.com/) archive, historical-forecast and forecast APIs, one point in Midtown (40.7549, −73.9840) | [CC BY 4.0](https://open-meteo.com/en/license) |
| US / New York public holidays | [`holidays`](https://github.com/vacanza/holidays) Python package | MIT |

Published forecasts and the track record are served from the Hugging Face dataset
[himel05/citibike-artifacts](https://huggingface.co/datasets/himel05/citibike-artifacts).

## References

- Ke, G. et al. (2017). [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html). NeurIPS 30. *(forecasting model)*
- Lundberg, S. M. and Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html). NeurIPS 30. *(SHAP explanations)*
- Taylor, S. J. and Letham, B. (2018). [Forecasting at Scale](https://doi.org/10.1080/00031305.2017.1380080). *The American Statistician* 72(1). *(Prophet comparison)*

*Trip and station data: Citi Bike System Data, used under the Citi Bike Data License Agreement; this project is
not affiliated with or endorsed by Citi Bike or Lyft. Weather data by Open-Meteo.com (CC BY 4.0).*
