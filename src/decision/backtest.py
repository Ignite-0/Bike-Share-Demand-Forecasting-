"""Backtest of empty/full alerts against real stockouts in the GBFS snapshots.

    python -m src.models.predict 2026-10-15 2026-10-29   # forecasts with archived-forecast weather (first)
    python -m src.decision.backtest

Origins: whole UTC hours with a snapshot within 5 min (not right after a feed gap). For each renting station:
actual empty = num_bikes_available = 0 in some snapshot in (t0, t0 + 3 h]; actual full = num_docks_available = 0.
A pair is dropped (not counted as "no stockout") when its window has a feed gap > 30 min, a non-renting snapshot,
or ends before t0 + 2 h 45 min. Alert levels are tuned by F1 on origins whose window ends before HOLDOUT, then
scored once on origins from HOLDOUT on. Overlapping windows make pairs correlated: treat counts as indicative.

Citi Bike's own rebalancing changes stock after t0, so some "false alarms" were averted stockouts: precision is a
lower bound. Forecast accuracy itself can't be checked until October trips are published (early November).
"""
import json

import duckdb
import numpy as np
import pandas as pd

from src.decision.rebalance import (DEFAULT_LEVELS, LOOKAHEAD_H, STATUS, THRESHOLDS, assess, info_sql,
                                    risk_scores)
from src.models.predict import FORECASTS, P90_SUM

HOLDOUT = pd.Timestamp("2026-10-26", tz="America/New_York").tz_convert("UTC")
GRID = np.arange(0, 8.5, 0.5)  # candidate alert levels
OUT = "reports/backtest_alerts.csv"


def pairs(status: str = STATUS) -> pd.DataFrame:
    return duckdb.sql(f"""
        WITH info AS ({info_sql()}),
        st AS (
            SELECT s.fetched_at, i.station_id, s.num_bikes_available AS bikes, s.num_docks_available AS docks,
                   s.is_renting = 1 AND s.is_installed = 1 AS ok, s.after_outage
            FROM '{status}' s JOIN info i ON i.gbfs_id = s.station_id
        ),
        snaps AS (SELECT DISTINCT fetched_at AS f, after_outage,
                         date_trunc('hour', fetched_at + INTERVAL 5 MINUTE) AS t0 FROM st),
        o AS (
            SELECT t0, arg_min(f, abs(epoch(f) - epoch(t0))) AS f FROM snaps
            WHERE abs(epoch(f) - epoch(t0)) <= 300 AND NOT after_outage GROUP BY t0
        ),
        base AS (
            SELECT o.t0, o.f, st.station_id, st.bikes, st.bikes + st.docks AS slots
            FROM o JOIN st ON st.fetched_at = o.f WHERE st.ok AND st.bikes + st.docks > 0
        )
        SELECT b.t0, b.station_id, b.bikes, b.slots,
               min(s.bikes) = 0 AS actual_empty, min(s.docks) = 0 AS actual_full
        FROM base b JOIN st s ON s.station_id = b.station_id
         AND s.fetched_at > b.f AND s.fetched_at <= b.f + INTERVAL {LOOKAHEAD_H} HOUR
        GROUP BY ALL
        HAVING NOT bool_or(s.after_outage) AND bool_and(s.ok)
           AND max(s.fetched_at) >= b.f + INTERVAL {LOOKAHEAD_H * 60 - 15} MINUTE
    """).df()


def prf(alert, actual) -> dict:
    tp = int((alert & actual).sum())
    p, r = tp / max(int(alert.sum()), 1), tp / max(int(actual.sum()), 1)
    return {"alerts": int(alert.sum()), "stockouts": int(actual.sum()), "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def main() -> None:
    p = pairs()
    fc = pd.read_parquet(FORECASTS)
    assert fc.ts_utc.min() <= p.t0.min() and fc.ts_utc.max() >= p.t0.max() + pd.Timedelta(hours=LOOKAHEAD_H - 1), \
        "forecasts don't cover the snapshots; rerun src.models.predict"
    with open(P90_SUM) as f:
        corr = json.load(f)
    a = assess(p, fc, corr)
    sc = risk_scores(a)
    tune = (a.t0 + pd.Timedelta(hours=LOOKAHEAD_H)) <= HOLDOUT
    hold = a.t0 >= HOLDOUT
    print(f"pairs: tune {tune.sum():,} ({a.t0[tune].nunique()} origins), hold-out {hold.sum():,} "
          f"({a.t0[hold].nunique()} origins, {a.t0[hold].min()} to {a.t0[hold].max()})")

    rows, best = [], {}
    for kind in ("empty", "full"):
        actual = a[f"actual_{kind}"]
        # stations not at risk now under the default level: what persistence can never catch
        new = sc[f"{kind}_persistence"] > DEFAULT_LEVELS[kind]
        for variant in ("point", "cautious", "persistence"):
            col = kind if variant == "point" else f"{kind}_{variant}"
            f1 = [prf(sc.loc[tune, col] <= lv, actual[tune])["f1"] for lv in GRID]
            best[col] = float(GRID[int(np.argmax(f1))])
            for name, lv in (("default", DEFAULT_LEVELS[kind]), ("tuned", best[col])):
                for split, m in (("tune", tune), ("holdout", hold), ("holdout_not_at_risk_now", hold & new)):
                    rows.append({"kind": kind, "variant": variant, "level_from": name, "level": lv, "split": split,
                                 **prf(sc.loc[m, col] <= lv, actual[m])})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, float_format="%.4f")
    with open(THRESHOLDS, "w") as f:
        json.dump({"levels": {k: best[k] for k in DEFAULT_LEVELS}, "persistence": {k: v for k, v in best.items()
                   if k.endswith("persistence")}, "tuned_on": f"GBFS origins with window ending before {HOLDOUT}",
                   "objective": "F1 per kind and variant"}, f, indent=1)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(out.to_string(index=False, float_format="%.3f"))
    print("tuned levels:", best)


if __name__ == "__main__":
    main()
