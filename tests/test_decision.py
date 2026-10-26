import glob
import json

import numpy as np
import pandas as pd

from src.decision.rebalance import assess, flag, moves, project, snapshot

T0 = pd.Timestamp("2026-10-20 12:00", tz="UTC")
CORR = {t: {str(h): 1.0 for h in range(1, 7)} for t in ("pickups", "dropoffs")}


def test_projection_clips_hour_by_hour():
    # steps -3, -5, +6 from 5 bikes: 2, 0 (clipped, not -3), then 6; and capped at 10 slots
    assert project([5], [[-3, -8, -2]], [10]).tolist() == [[2, 0, 6]]
    assert project([9], [[4, 4]], [10]).tolist() == [[10, 10]]


def _stations(bikes, slots, pu, dr, pu90=None, dr90=None):
    snap = pd.DataFrame({"station_id": [f"s{i}" for i in range(len(bikes))], "t0": T0, "bikes": bikes, "slots": slots})
    fc = pd.DataFrame([{"station_id": s, "ts_utc": T0 + pd.Timedelta(hours=h), "pu_mean": p, "dr_mean": d,
                        "pu_p90": p if pu90 is None else pu90[i], "dr_p90": d if dr90 is None else dr90[i]}
                       for i, (s, p, d) in enumerate(zip(snap.station_id, pu, dr)) for h in range(6)])
    return assess(snap, fc, CORR)


def test_alert_levels_and_cautious_variant():
    lv = {"empty": 2, "full": 2, "empty_cautious": 2, "full_cautious": 2}
    # s0: 5 bikes, losing 1/h -> 2 bikes after 3 h: empty at level 2. s1: 4 bikes, flat: no alert, but its P90
    # pick-ups (2/h) empty it. s2: 17 of 20 slots, +1/h -> 0 free docks: full.
    a = flag(_stations([5, 4, 17], [20, 20, 20], pu=[1, 0, 0], dr=[0, 0, 1], pu90=[1, 2, 0]), lv)
    assert a["empty"].tolist() == [True, False, False]
    assert a["empty_cautious"].tolist() == [True, True, False]
    assert a["full"].tolist() == [False, False, True]
    assert (a["min_empty90"] <= a["min_point"]).all() and (a["max_full90"] >= a["max_point"]).all()
    # a station with no forecast keeps its stock
    lone = assess(pd.DataFrame({"station_id": ["new"], "t0": [T0], "bikes": [3], "slots": [10]}),
                  pd.DataFrame(columns=["station_id", "ts_utc", "pu_mean", "pu_p90", "dr_mean", "dr_p90"]), CORR)
    assert lone["min_point"].item() == 3 and lone["bikes_6h"].item() == 3


def test_moves_stay_in_band():
    lv = {"empty": 2, "full": 2, "empty_cautious": 2, "full_cautious": 2}
    a = flag(_stations([1, 15, 19], [20, 20, 20], pu=[0, 0, 0], dr=[0, 0, 0]), lv)
    a = a.assign(name=a.station_id, lat=[40.70, 40.701, 40.80], lng=-74.0)  # s2 is ~11 km away
    m = moves(a)
    assert m[["from", "to", "bikes"]].values.tolist() == [["s1", "s0", 5]]  # s0 up to 30% (6), s1 stays >= 6
    a2 = a.set_index("station_id")
    assert a2.at["s1", "min_point"] - 5 >= 0.3 * 20 and a2.at["s0", "max_point"] + 5 <= 0.7 * 20


def test_snapshot_drops_non_renting(tmp_path):
    info = json.load(open(sorted(glob.glob("data/raw/gbfs/info/*.json"))[-1]))[:2]
    ts = pd.Timestamp("2026-10-20 12:00", tz="UTC")
    pd.DataFrame({"fetched_at": ts, "station_id": [s["station_id"] for s in info], "num_bikes_available": 3,
                  "num_docks_available": 5, "is_renting": [1, 0], "is_installed": 1}).to_parquet(tmp_path / "s.parquet")
    snap = snapshot(ts, str(tmp_path / "s.parquet").replace("\\", "/"))
    assert len(snap) == 1 and snap["slots"].item() == 8 and np.isfinite(snap["lat"]).all()
