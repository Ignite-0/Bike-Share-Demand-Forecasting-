"""Metrics, baselines and the experiment log.

Usage: python -m src.models.evaluate   # scores both baselines on val, appends to reports/experiments.csv
"""
import csv
import json
import os
import subprocess
import time
from datetime import datetime

import numpy as np
import pandas as pd
import psutil

LOG = "reports/experiments.csv"
LOG_COLS = ["run_id", "timestamp", "target", "feature_set", "loss", "library", "device", "params", "train_rows",
            "train_s", "predict_s", "peak_ram_gb", "peak_vram_gb", "val_mae", "val_rmse", "val_wape",
            "val_bias", "best_iter", "git_commit"]
PREFIX = {"pickups": "pu", "dropoffs": "dr"}


def metrics(y, pred) -> dict:
    y, pred = np.asarray(y, dtype="float64"), np.asarray(pred, dtype="float64")
    err = pred - y
    return {"mae": np.abs(err).mean(), "rmse": np.sqrt((err ** 2).mean()),
            "wape": np.abs(err).sum() / y.sum(), "bias": err.sum() / y.sum()}


def breakdown(df: pd.DataFrame, y: str, pred: str, by: str) -> pd.DataFrame:
    return df.groupby(by, dropna=False, observed=True).apply(
        lambda g: pd.Series({**metrics(g[y], g[pred]), "rows": len(g)}), include_groups=False)


def baseline(df: pd.DataFrame, target: str, kind: str) -> pd.Series:
    """B-fresh: lag168 -> same_hour_4w -> 0 (needs last week's trips). B-published: recent_profile -> profile -> 0."""
    t = PREFIX[target]
    first, second = {"fresh": ("lag168", "same_hour_4w"), "published": ("recent_profile", "profile")}[kind]
    return df[f"{t}_{first}"].fillna(df[f"{t}_{second}"]).fillna(0)


def log_run(row: dict) -> None:
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "git_commit": git,
           "peak_ram_gb": round(psutil.Process().memory_info().peak_wset / 2 ** 30, 2), **row}
    row["run_id"] = f"{row['timestamp'].replace(':', '')}_{row['target']}_{row['feature_set']}_{row['library']}"
    row = {k: round(v, 5) if isinstance(v, float) else json.dumps(v) if isinstance(v, dict) else v
           for k, v in row.items()}
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, LOG_COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    print({k: row.get(k) for k in ("target", "feature_set", "loss", "library", "val_mae", "val_wape", "val_bias",
                                   "train_s", "peak_ram_gb", "best_iter")})


if __name__ == "__main__":
    from src.models.data import load

    for target in PREFIX:
        t = PREFIX[target]
        cols = [target, f"{t}_lag168", f"{t}_same_hour_4w", f"{t}_recent_profile", f"{t}_profile"]
        df = load("val", cols)
        for kind in ("fresh", "published"):
            t0 = time.time()
            pred = baseline(df, target, kind)
            m = metrics(df[target], pred)
            log_run({"target": target, "feature_set": f"B-{kind}", "loss": "", "library": "baseline",
                     "device": "cpu", "params": {}, "train_rows": 0, "train_s": 0.0,
                     "predict_s": time.time() - t0, **{f"val_{k}": v for k, v in m.items()}})
