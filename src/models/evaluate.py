"""Metrics, baselines and the experiment log.

Usage:
    python -m src.models.evaluate           # scores both baselines on val, appends to reports/experiments.csv
    python -m src.models.evaluate --errors pickups_A+sid_tweedie1.2_lightgbm_cpu_tuned
        # step 6: breakdowns, worst station-days and monthly error for a saved run -> reports/errors/
    python -m src.models.evaluate --blend pickups_A+sid_tweedie1.2_lightgbm_cpu_tuned   # hour-weighted blend
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


def _groups(df: pd.DataFrame) -> pd.DataFrame:
    """Reporting dimensions of the plan (section 4). NULL cluster = opened after June ("new")."""
    names = {0: "0 residential", 1: "1 evening-heavy", 2: "2 balanced", 3: "3 business", -1: "-1 low volume"}
    return pd.DataFrame({
        "cluster": df["cluster"].map(names).fillna("new"),
        "volume_tier": df["volume_tier"].fillna("new"),
        "hour": df["hour"],
        "daytype": np.select([df["is_holiday"], df["is_weekend"]], ["holiday", "weekend"], "weekday"),
        "precip": pd.cut(df["precip_mm"], [-1, 0.1, 0.5, 2, 1e9], right=False,
                         labels=["dry", "0.1-0.5", "0.5-2", ">=2"]).astype(str),
    }, index=df.index)


def compare(df: pd.DataFrame, y: str, by: str) -> pd.DataFrame:
    """Model vs both baselines for one dimension; gain_* = MAE reduction vs that baseline."""
    out = breakdown(df, y, "pred", by)
    for b in ("fresh", "published"):
        out[f"mae_{b}"] = breakdown(df, y, f"b_{b}", by)["mae"]
        out[f"gain_{b}"] = 1 - out["mae"] / out[f"mae_{b}"]
    return out


def error_report(name: str, out: str = "reports/errors") -> None:
    """Breakdowns by the plan's dimensions, 10 worst station-days, and error by month (H5) for a saved run."""
    import duckdb
    import lightgbm as lgb

    from src.models.data import FEATURES, load

    target = name.split("_")[0]
    t = PREFIX[target]
    base = [f"{t}_{c}" for c in ("lag168", "same_hour_4w", "recent_profile", "profile")]
    rep_cols = ["cluster", "volume_tier", "hour", "is_holiday", "is_weekend", "precip_mm"]
    df = duckdb.sql(f"""
        SELECT p.station_id, p.ts_utc, f.ts_local, p.{target}, p.pred, {", ".join("f." + c for c in base + rep_cols)}
        FROM 'data/processed/preds/{name}.parquet' p JOIN {FEATURES} f USING (station_id, ts_utc)
        WHERE f.split = 'val'""").df()
    df["b_fresh"], df["b_published"] = baseline(df, target, "fresh"), baseline(df, target, "published")
    df = df.join(_groups(df), rsuffix="_g")
    os.makedirs(out, exist_ok=True)

    tables = []
    for dim in ("cluster", "volume_tier", "daytype", "precip", "hour"):
        by = dim if dim in ("daytype", "precip") else f"{dim}_g"
        tables.append(compare(df, target, by).reset_index(names="group").assign(dim=dim))
    tab = pd.concat(tables)[["dim", "group", "rows", "mae", "rmse", "wape", "bias", "mae_fresh", "mae_published",
                             "gain_fresh", "gain_published"]]
    tab.to_csv(f"{out}/{name}_breakdown.csv", index=False, float_format="%.4f")

    df["abs_err"] = (df["pred"] - df[target]).abs()
    df["date"] = df["ts_local"].dt.date
    worst = (df.groupby(["station_id", "date"], observed=True)
             .agg(abs_err=("abs_err", "sum"), actual=(target, "sum"), pred=("pred", "sum"),
                  b_published=("b_published", "sum"), cluster=("cluster_g", "first"),
                  precip_mm=("precip_mm", "sum"))
             .nlargest(10, "abs_err").reset_index())
    worst.to_csv(f"{out}/{name}_worst_days.csv", index=False, float_format="%.2f")
    del df

    # H5: the same model on every month it was fit on, early-stopped on (Jun) and scored on (Jul-Aug)
    model = lgb.Booster(model_file=f"data/processed/models/{name}.txt")
    rows = []
    for split in ("train", "val"):
        m = load(split, model.feature_name() + [target, "ts_local"] + base)
        m["pred"] = model.predict(m[model.feature_name()])
        m["b_fresh"], m["b_published"] = baseline(m, target, "fresh"), baseline(m, target, "published")
        m["month"] = m["ts_local"].dt.strftime("%Y-%m")
        rows.append(compare(m, target, "month"))
        del m
    monthly = pd.concat(rows).reset_index(names="month")
    monthly["role"] = np.select([monthly["month"] < "2026-06", monthly["month"] == "2026-06"],
                                ["fit", "early stop"], "val")
    monthly.to_csv(f"{out}/{name}_monthly.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tab.to_string(index=False, float_format="%.3f"), worst.to_string(index=False),
              monthly.to_string(index=False, float_format="%.3f"), sep="\n\n")


def fit_blend(df: pd.DataFrame, y: str) -> pd.Series:
    """Per-hour weight w on the model in w * pred + (1 - w) * b_published, minimizing MAE (grid of 0.05)."""
    grid = np.linspace(0, 1, 21)
    return df.groupby("hour").apply(lambda g: grid[np.argmin(
        [np.abs(w * g["pred"] + (1 - w) * g["b_published"] - g[y]).mean() for w in grid])], include_groups=False)


def blend_report(name: str, out: str = "reports/errors") -> None:
    """Hour-weighted blend of a saved model with B-published. Weights fit on June (protocol: score Jul-Aug) and on
    July (score Aug only, as with a monthly refit). Logs both to the experiment log."""
    import lightgbm as lgb

    from src.models.data import STOP_FROM, load

    target, feature_set = name.split("_")[:2]
    t = PREFIX[target]
    model = lgb.Booster(model_file=f"data/processed/models/{name}.txt")
    cols = model.feature_name() + [target, "ts_local", "hour", f"{t}_recent_profile", f"{t}_profile"]

    def scored(split, where):
        d = load(split, cols, where=where)
        d["pred"], d["b_published"] = model.predict(d[model.feature_name()]), baseline(d, target, "published")
        return d[[target, "ts_local", "hour", "pred", "b_published"]]

    june, val = scored("train", f"AND ts_local >= '{STOP_FROM}'"), scored("val", "")
    july, aug = val[val.ts_local < "2026-08-01"], val[val.ts_local >= "2026-08-01"]
    weights = pd.DataFrame({"fit_jun": fit_blend(june, target), "fit_jul": fit_blend(july, target)})
    weights.to_csv(f"{out}/{name}_blend_weights.csv")
    for fit, score, label in (("fit_jun", val, "jun"), ("fit_jul", aug, "jul")):
        w = score["hour"].map(weights[fit])
        blend = w * score["pred"] + (1 - w) * score["b_published"]
        m = metrics(score[target], blend)
        print(label, "model", {k: round(v, 4) for k, v in metrics(score[target], score["pred"]).items()},
              "\n   ", label, "blend", {k: round(v, 4) for k, v in m.items()},
              "\n   ", label, "B-pub", {k: round(v, 4) for k, v in metrics(score[target], score["b_published"]).items()})
        log_run({"target": target, "feature_set": f"{feature_set}+blend" + ("" if label == "jun" else "@aug-only"),
                 "loss": "", "library": "blend", "device": "cpu", "params": {"weights_fit_on": label, "model": name},
                 "train_rows": len(june if label == "jun" else july), "train_s": 0.0, "predict_s": 0.0,
                 **{f"val_{k}": v for k, v in m.items()}})
    print(weights.T.to_string(float_format="%.2f"))


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
    import sys

    from src.models.data import load

    if len(sys.argv) == 3 and sys.argv[1] == "--errors":
        error_report(sys.argv[2])
        sys.exit()
    if len(sys.argv) == 3 and sys.argv[1] == "--blend":
        blend_report(sys.argv[2])
        sys.exit()

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
