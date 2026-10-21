"""Random search for LightGBM on one fold, one process per trial (so peak RAM is per trial).

Trials are logged to reports/experiments.csv like any run; this prints the ranking at the end.
    python -m src.models.tune --target pickups --features A+sid --loss tweedie1.2 --trials 20 --minutes 95
"""
import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd

from src.models.evaluate import LOG


def sample(rng: random.Random) -> dict:
    return {"learning_rate": round(10 ** rng.uniform(-1.5, -0.8), 4),  # 0.03 .. 0.16
            "num_leaves": rng.choice([31, 63, 127, 255]),
            "min_data_in_leaf": rng.choice([20, 50, 100, 200, 500]),
            "feature_fraction": round(rng.uniform(0.6, 1.0), 2),
            "bagging_fraction": round(rng.uniform(0.6, 1.0), 2), "bagging_freq": 1,
            "lambda_l2": rng.choice([0, 0.1, 1, 10])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="pickups")
    ap.add_argument("--features", default="A+sid")
    ap.add_argument("--loss", default="tweedie1.2")
    ap.add_argument("--fold", default="val")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=95, help="stop starting new trials after this")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng, start = random.Random(a.seed), time.time()
    since = datetime.now().isoformat(timespec="seconds")
    for i in range(a.trials):
        if time.time() - start > a.minutes * 60:
            print(f"timebox reached after {i} trials")
            break
        p = sample(rng)
        print(f"trial {i}: {p}", flush=True)
        subprocess.run([sys.executable, "-m", "src.models.train", "--target", a.target, "--features", a.features,
                        "--loss", a.loss, "--fold", a.fold, "--params", json.dumps(p), "--no-preds"],
                       stdout=subprocess.DEVNULL, check=True)
    d = pd.read_csv(LOG)
    d = d[(d.timestamp >= since) & (d.target == a.target)].sort_values("val_mae")
    print(d[["val_mae", "val_wape", "val_bias", "train_s", "predict_s", "peak_ram_gb", "best_iter", "params"]]
          .to_string(index=False, max_colwidth=200))
