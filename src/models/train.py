"""Train one model, score it on val, append the run to reports/experiments.csv.

Fits on Oct-May, early-stops on June (the last training month), scores Jul-Aug. Run each experiment in its
own process so peak RAM is per run:
    python -m src.models.train --target pickups --features A+C --loss poisson
Feature sets: tiers joined by '+', plus 'sid' for station_id as a categorical (e.g. A+C+sid).
Tier B is an upper bound only (PRODUCTION_TIERS); never report it as production performance.
"""
import argparse
import os
import time

import lightgbm as lgb

from src.models.data import STOP_FROM, feature_cols, load
from src.models.evaluate import log_run, metrics

PREDS = "data/processed/preds"
# (split, where) for fit / early-stop / score. "val" is the default protocol; "aug" is the rolling fold that
# has seen the flat Jul plateau: fit Oct-Jul, stop on Aug 1-15, score Aug 16-31.
FOLDS = {
    "val": [("train", f"AND ts_local < '{STOP_FROM}'"), ("train", f"AND ts_local >= '{STOP_FROM}'"), ("val", "")],
    "aug": [("train,val", "AND ts_local < '2026-08-01'"),
            ("val", "AND ts_local >= '2026-08-01' AND ts_local < '2026-08-16'"),
            ("val", "AND ts_local >= '2026-08-16'")],
}


def lgb_objective(loss: str) -> dict:
    """'poisson', 'l2', 'l1', 'tweedie1.5' -> LightGBM objective params."""
    if loss.startswith("tweedie"):
        return {"objective": "tweedie", "tweedie_variance_power": float(loss[7:])}
    return {"objective": loss}


def train(target: str, feature_set: str, loss: str = "poisson", params: dict | None = None, fold: str = "val"):
    feats = feature_cols(feature_set)
    params = {"learning_rate": 0.1, "num_threads": 16, "verbose": -1, **lgb_objective(loss), **(params or {})}
    (fit_s, fit_w), (stop_s, stop_w), (score_s, score_w) = FOLDS[fold]
    fit = load(fit_s, feats + [target], where=fit_w)
    stop = load(stop_s, feats + [target], where=stop_w)
    n = len(fit)
    t0 = time.time()
    dfit = lgb.Dataset(fit[feats], fit[target], free_raw_data=True)
    dstop = lgb.Dataset(stop[feats], stop[target], reference=dfit)
    del fit, stop
    model = lgb.train(params, dfit, num_boost_round=2000, valid_sets=[dstop],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    train_s = time.time() - t0
    del dfit, dstop

    val = load(score_s, feats + [target, "station_id", "ts_utc"], where=score_w)
    if fold != "val":
        feature_set = f"{feature_set}@{fold}"  # logged name; val_* columns then hold this fold's scores
    t0 = time.time()
    val["pred"] = model.predict(val[feats], num_iteration=model.best_iteration)
    predict_s = time.time() - t0
    os.makedirs(PREDS, exist_ok=True)
    val[["station_id", "ts_utc", target, "pred"]].to_parquet(f"{PREDS}/{target}_{feature_set}_{loss}_lightgbm.parquet")
    m = metrics(val[target], val["pred"])
    log_run({"target": target, "feature_set": feature_set, "loss": loss, "library": "lightgbm", "device": "cpu",
             "params": params, "train_rows": n, "train_s": train_s, "predict_s": predict_s,
             "best_iter": model.best_iteration, **{f"val_{k}": v for k, v in m.items()}})
    return model, m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["pickups", "dropoffs"])
    ap.add_argument("--features", required=True)
    ap.add_argument("--loss", default="poisson")
    ap.add_argument("--fold", default="val", choices=list(FOLDS))
    a = ap.parse_args()
    train(a.target, a.features, a.loss, fold=a.fold)
