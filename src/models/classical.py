"""Step 8: Prophet per station on the 20 busiest stations (pick-ups), scored on the same val rows as LightGBM.

Each station gets its own Prophet fit on Oct-Jun (all train months; Prophet has no early stopping): weekly
seasonality, separate daily shapes for weekdays and off-days (weekend or holiday), US holidays, temperature and
precipitation as regressors (observed weather, as in the LightGBM val runs). No yearly seasonality: 9 months
can't identify it. Variants: linear vs flat trend, additive vs multiplicative seasonality.
    python -m src.models.classical
"""
import itertools
import logging
import time

import numpy as np
import pandas as pd

from src.models.data import load
from src.models.evaluate import baseline, log_run, metrics

N_STATIONS = 20
LGBM = "pickups_A+sid_tweedie1.2_lightgbm_cpu_tuned"
OUT = "reports/errors/prophet_top20.csv"


def fit_predict(train: pd.DataFrame, future: pd.DataFrame, growth: str, mode: str) -> np.ndarray:
    from prophet import Prophet

    m = Prophet(growth=growth, seasonality_mode=mode, yearly_seasonality=False, daily_seasonality=False)
    for name in ("weekday", "offday"):
        m.add_seasonality(f"daily_{name}", period=1, fourier_order=10, condition_name=f"on_{name}")
    m.add_country_holidays("US")
    for r in ("temp_c", "precip_mm"):
        m.add_regressor(r)
    m.fit(train)
    out = m.predict(future)
    assert (out["ds"].to_numpy() == future["ds"].to_numpy()).all(), "future must be sorted by ds"
    return out["yhat"].clip(lower=0).to_numpy()


def frame(d: pd.DataFrame) -> pd.DataFrame:
    off = d["is_weekend"] | d["is_holiday"]
    return pd.DataFrame({"ds": d["ts_local"], "y": d["pickups"], "temp_c": d["temp_c"], "precip_mm": d["precip_mm"],
                         "on_weekday": ~off, "on_offday": off})


def main() -> None:
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    top = load("train", ["station_id", "pickups"]).groupby("station_id", observed=True)["pickups"].sum()
    top = list(top.nlargest(N_STATIONS).index)  # busiest by training months only, no val peeking
    ids = ", ".join(f"'{s}'" for s in top)
    base = ["pu_lag168", "pu_same_hour_4w", "pu_recent_profile", "pu_profile"]
    cols = ["station_id", "ts_utc", "ts_local", "hour", "pickups", "temp_c", "precip_mm", "is_weekend", "is_holiday"]
    # sorted by time within station: Prophet.predict returns rows ordered by ds, assigned back positionally
    key = ["station_id", "ts_utc"]
    train = load("train", cols, where=f"AND station_id IN ({ids})").sort_values(key, ignore_index=True)
    val = load("val", cols + base, where=f"AND station_id IN ({ids})").sort_values(key, ignore_index=True)

    lgbm = pd.read_parquet(f"data/processed/preds/{LGBM}.parquet")[["station_id", "ts_utc", "pred"]]
    lgbm["station_id"] = lgbm["station_id"].astype(str)
    val["station_id"] = val["station_id"].astype(str)
    val = val.merge(lgbm, on=["station_id", "ts_utc"], how="left")
    w = val["hour"].map(pd.read_csv(f"reports/errors/{LGBM}_blend_weights.csv", index_col="hour")["fit_jun"])
    val["b_fresh"], val["b_published"] = baseline(val, "pickups", "fresh"), baseline(val, "pickups", "published")
    val["blend"] = w * val["pred"] + (1 - w) * val["b_published"]
    train["station_id"] = train["station_id"].astype(str)

    variants = list(itertools.product(("linear", "flat"), ("additive", "multiplicative")))
    for growth, mode in variants:
        col, t0 = f"prophet_{growth}_{mode}", time.time()
        for s in top:
            rows = val["station_id"] == s
            val.loc[rows, col] = fit_predict(frame(train[train.station_id == s]), frame(val[rows]), growth, mode)
        m = metrics(val["pickups"], val[col])
        log_run({"target": "pickups", "feature_set": "top20", "loss": "", "library": "prophet", "device": "cpu",
                 "params": {"growth": growth, "seasonality_mode": mode, "stations": N_STATIONS},
                 "train_rows": int(train.shape[0]), "train_s": time.time() - t0, "predict_s": 0.0,
                 **{f"val_{k}": v for k, v in m.items()}})

    methods = ["pred", "blend", "b_published", "b_fresh"] + [f"prophet_{g}_{m}" for g, m in variants]
    overall = pd.DataFrame({k: metrics(val["pickups"], val[k]) for k in methods}).T
    per_station = val.groupby("station_id")[["pickups"] + methods].apply(
        lambda g: pd.Series({k: (g[k] - g["pickups"]).abs().mean() for k in methods}))
    overall.assign(stations_won=per_station.idxmin(axis=1).value_counts()).fillna(0).to_csv(OUT, float_format="%.4f")
    per_station.to_csv(OUT.replace(".csv", "_by_station.csv"), float_format="%.3f")
    print(overall.to_string(float_format="%.3f"), per_station.idxmin(axis=1).value_counts().to_string(), sep="\n\n")


if __name__ == "__main__":
    main()
