"""Train one model, score it on val, append the run to reports/experiments.csv.

Fits on Oct-May, early-stops on June (the last training month), scores Jul-Aug. Run each experiment in its
own process so peak RAM is per run:
    python -m src.models.train --target pickups --features A+C --loss poisson
    python -m src.models.train --target pickups --features A+sid --loss poisson --library catboost --device gpu
Feature sets: tiers joined by '+', plus 'sid' for station_id as a categorical (e.g. A+C+sid).
Tier B is an upper bound only (PRODUCTION_TIERS); never report it as production performance.
"""
import argparse
import contextlib
import os
import subprocess
import threading
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


def cb_loss(loss: str) -> str:
    """Same names -> CatBoost loss_function. CatBoost 1.2.10 crashes on Tweedie with task_type=GPU."""
    if loss.startswith("tweedie"):
        return f"Tweedie:variance_power={loss[7:]}"
    return {"poisson": "Poisson", "l2": "RMSE", "l1": "MAE"}[loss]


def gpu_mem_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


class VramPeak:
    """Polls nvidia-smi once a second; .gb = peak used above the level at start (whole device, so approximate)."""

    def __enter__(self):
        self.base = self.peak = gpu_mem_mb()
        self.done = threading.Event()
        self.t = threading.Thread(target=self._poll, daemon=True)
        self.t.start()
        return self

    def _poll(self):
        while not self.done.wait(1):
            self.peak = max(self.peak, gpu_mem_mb())

    def __exit__(self, *_):
        self.done.set()
        self.t.join()
        self.gb = (self.peak - self.base) / 1024


def fit_lightgbm(fit, stop, feats, target, loss, params):
    params = {"learning_rate": 0.1, "num_threads": 16, "verbose": -1, **lgb_objective(loss), **(params or {})}
    dfit = lgb.Dataset(fit[feats], fit[target], free_raw_data=True)
    dstop = lgb.Dataset(stop[feats], stop[target], reference=dfit)
    model = lgb.train(params, dfit, num_boost_round=4000, valid_sets=[dstop],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    return model, params, model.best_iteration, lambda x: model.predict(x, num_iteration=model.best_iteration)


def fit_catboost(fit, stop, feats, target, loss, params, device):
    from catboost import CatBoostRegressor, Pool

    params = {"learning_rate": 0.1, "iterations": 4000, "early_stopping_rounds": 50, "loss_function": cb_loss(loss),
              "task_type": device.upper(), "thread_count": 16, "verbose": 100,
              "train_dir": "data/catboost_info", **(params or {})}
    cats = [c for c in feats if c == "station_id"]
    model = CatBoostRegressor(**params)
    model.fit(Pool(fit[feats], fit[target], cat_features=cats), eval_set=Pool(stop[feats], stop[target], cat_features=cats))
    return model, params, model.get_best_iteration(), model.predict


def train(target: str, feature_set: str, loss: str = "poisson", params: dict | None = None, fold: str = "val",
          library: str = "lightgbm", device: str = "cpu"):
    feats = feature_cols(feature_set)
    (fit_s, fit_w), (stop_s, stop_w), (score_s, score_w) = FOLDS[fold]
    fit = load(fit_s, feats + [target], where=fit_w)
    stop = load(stop_s, feats + [target], where=stop_w)
    n = len(fit)
    t0 = time.time()
    with VramPeak() if device == "gpu" else contextlib.nullcontext() as vram:
        if library == "lightgbm":
            model, params, best_iter, predict = fit_lightgbm(fit, stop, feats, target, loss, params)
        else:
            model, params, best_iter, predict = fit_catboost(fit, stop, feats, target, loss, params, device)
    train_s = time.time() - t0
    del fit, stop

    val = load(score_s, feats + [target, "station_id", "ts_utc"], where=score_w)
    if fold != "val":
        feature_set = f"{feature_set}@{fold}"  # logged name; val_* columns then hold this fold's scores
    t0 = time.time()
    val["pred"] = predict(val[feats])
    predict_s = time.time() - t0
    os.makedirs(PREDS, exist_ok=True)
    val[["station_id", "ts_utc", target, "pred"]].to_parquet(f"{PREDS}/{target}_{feature_set}_{loss}_{library}_{device}.parquet")
    m = metrics(val[target], val["pred"])
    log_run({"target": target, "feature_set": feature_set, "loss": loss, "library": library, "device": device,
             "params": params, "train_rows": n, "train_s": train_s, "predict_s": predict_s, "best_iter": best_iter,
             "peak_vram_gb": round(vram.gb, 2) if device == "gpu" else "", **{f"val_{k}": v for k, v in m.items()}})
    return model, m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["pickups", "dropoffs"])
    ap.add_argument("--features", required=True)
    ap.add_argument("--loss", default="poisson")
    ap.add_argument("--fold", default="val", choices=list(FOLDS))
    ap.add_argument("--library", default="lightgbm", choices=["lightgbm", "catboost"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    a = ap.parse_args()
    if a.library == "lightgbm" and a.device == "gpu":
        ap.error("LightGBM GPU needs a source build; it runs on CPU only here")
    train(a.target, a.features, a.loss, fold=a.fold, library=a.library, device=a.device)
