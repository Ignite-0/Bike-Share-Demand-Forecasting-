"""Step 10: freeze the final config, then the single test run.

    python -m src.models.final prepare   # val months only: Oct-Jul ("aug" fold) models -> tree counts and
                                         # blend weights fit on August -> models/final_config.json
    python -m src.models.final test      # THE single test run: retrain Oct-Aug with the frozen config, score
                                         # September once. Refuses to run twice.

Final config (reports/PHASE5_REPORT.md): LightGBM on A+sid with the tuned parameters; Tweedie 1.2 mean model
blended per hour with B-published; quantile P90 as the cautious upper bound for Phase 6.
"""
import json
import os
import subprocess
import sys
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.build import TARGETS
from src.models.data import feature_cols, load
from src.models.evaluate import PREFIX, baseline, compare, fit_blend, metrics, _groups
from src.models.train import TUNED, lgb_objective, train

FEATURE_SET = "A+sid"
LOSSES = ("tweedie1.2", "q90")  # mean model (blended) and cautious upper bound
CONFIG = "models/final_config.json"
RESULTS = "reports/test_results.csv"
DEV = "data/processed/models"


def prepare() -> None:
    """Everything the test run needs, from train + val only."""
    feats = feature_cols(FEATURE_SET)
    cfg = {"feature_set": FEATURE_SET, "features": feats, "params": TUNED, "train_window_local": ["2025-10-01", "2026-09-01"],
           "models": {}, "blend_weights": {},
           "notes": "num_boost_round = best iteration of the same config fit on Oct-Jul and early-stopped on Aug 1-15 "
                    "(not scaled); blend weights fit on all of August with that Oct-Jul Tweedie model."}
    for target in TARGETS:
        t = PREFIX[target]
        for loss in LOSSES:
            name = f"{target}_{FEATURE_SET}@aug_{loss}_lightgbm_cpu_tuned"
            if not os.path.exists(f"{DEV}/{name}.txt"):
                # ponytail: in-process, so this run's peak RAM is logged cumulatively; fine for a one-off
                train(target, FEATURE_SET, loss, TUNED, fold="aug", tag="tuned")
            booster = lgb.Booster(model_file=f"{DEV}/{name}.txt")
            cfg["models"][f"{target}_{loss}"] = {"target": target, "loss": loss, **lgb_objective(loss),
                                                 "num_boost_round": booster.current_iteration(), "aug_fold_model": name}
            if loss == "tweedie1.2":
                aug = load("val", feats + [target, f"{t}_recent_profile", f"{t}_profile"],
                           where="AND ts_local >= '2026-08-01'")
                aug["pred"], aug["b_published"] = booster.predict(aug[feats]), baseline(aug, target, "published")
                cfg["blend_weights"][target] = {int(h): float(w) for h, w in fit_blend(aug, target).items()}
    cfg["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    os.makedirs("models", exist_ok=True)
    with open(CONFIG, "w") as f:
        json.dump(cfg, f, indent=1)
    print(json.dumps({k: v for k, v in cfg.items() if k != "features"}, indent=1))


def run_test() -> None:
    if os.path.exists(RESULTS):
        sys.exit(f"{RESULTS} exists: the test month has already been scored once. Not running again.")
    with open(CONFIG) as f:
        cfg = json.load(f)
    feats, stamp, rows = cfg["features"], date.today().strftime("%Y%m%d"), []
    for target in TARGETS:
        t = PREFIX[target]
        fit = load("train,val", feats + [target])  # Oct-Aug
        models = {}
        for loss in LOSSES:
            m = cfg["models"][f"{target}_{loss}"]
            params = {**cfg["params"], **lgb_objective(loss), "num_threads": 16, "verbose": -1}
            models[loss] = lgb.train(params, lgb.Dataset(fit[feats], fit[target]), num_boost_round=m["num_boost_round"])
            models[loss].save_model(f"models/{target}_{loss}_lightgbm_{stamp}.txt")
        n_fit = len(fit)
        del fit

        base = [f"{t}_{c}" for c in ("lag168", "same_hour_4w", "recent_profile", "profile")]
        test = load("test", feats + [target, "is_storm", "cluster", "volume_tier", "is_holiday", "is_weekend",
                                     "precip_mm"] + base, final=True)
        test["model"] = models["tweedie1.2"].predict(test[feats])
        test["p90"] = models["q90"].predict(test[feats])
        test["b_fresh"], test["b_published"] = baseline(test, target, "fresh"), baseline(test, target, "published")
        w = test["hour"].map({int(h): v for h, v in cfg["blend_weights"][target].items()})
        test["pred"] = w * test["model"] + (1 - w) * test["b_published"]

        for subset, df in (("all", test), ("without_storm", test[~test["is_storm"]])):
            for method in ("pred", "model", "b_published", "b_fresh"):
                rows.append({"target": target, "subset": subset, "method": method, "rows": len(df),
                             **metrics(df[target], df[method]),
                             "gain_vs_fresh": 1 - (df[method] - df[target]).abs().mean()
                             / (df["b_fresh"] - df[target]).abs().mean(),
                             "gain_vs_published": 1 - (df[method] - df[target]).abs().mean()
                             / (df["b_published"] - df[target]).abs().mean(),
                             "above_p90": float((df[target] > df["p90"]).mean()) if method == "pred" else np.nan})
        test = test.join(_groups(test), rsuffix="_g")
        compare(test, target, "cluster_g").to_csv(f"reports/errors/test_{target}_by_cluster.csv", float_format="%.4f")
        meta = {**{k: cfg[k] for k in ("feature_set", "features", "params", "train_window_local", "git_commit")},
                "models": {k: v for k, v in cfg["models"].items() if v["target"] == target},
                "blend_weights": cfg["blend_weights"][target], "train_rows": n_fit,
                "test": [r for r in rows if r["target"] == target]}
        with open(f"models/{target}_meta.json", "w") as f:
            json.dump(meta, f, indent=1, default=float)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS, index=False, float_format="%.4f")
    print(out.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    {"prepare": prepare, "test": run_test}[sys.argv[1]]()
