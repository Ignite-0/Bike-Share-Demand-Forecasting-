"""API on a stubbed GBFS feed and synthetic forecasts: no models, data/processed or network needed."""
import json

import pandas as pd
from fastapi.testclient import TestClient

import src.api.main as api
import src.decision.rebalance as rebalance

NOW = pd.Timestamp.now(tz="UTC")
STATUS = {"last_updated": int(NOW.timestamp()), "data": {"stations": [
    # A: 1 bike, losing 2/h -> empty. B: 10 of 20, flat. C: 19 of 20 -> full.
    {"station_id": g, "num_bikes_available": b, "num_docks_available": 20 - b, "is_renting": 1, "is_installed": 1}
    for g, b in (("ga", 1), ("gb", 10), ("gc", 19))]}}
INFO = {"data": {"stations": [
    {"station_id": g, "short_name": s, "name": n, "lat": 40.75 + i * 0.001, "lon": -73.98}
    for i, (g, s, n) in enumerate((("ga", "5000.01", "A"), ("gb", "5000.02", "B"), ("gc", "5000.03", "C")))]}}


def client(tmp_path, monkeypatch):
    hours = pd.date_range(NOW.floor("h"), periods=12, freq="h")
    flow = {"5000.01": (2, 0), "5000.02": (0, 0), "5000.03": (0, 2)}
    api.state["fc"] = pd.DataFrame([{"station_id": s, "ts_utc": t, "hour": t.hour, "pu_mean": pu, "pu_p90": pu,
                                     "dr_mean": dr, "dr_p90": dr} for s, (pu, dr) in flow.items() for t in hours])
    api.state["card"] = {"built_at": str(NOW), "summary": []}
    api.state["track"] = pd.DataFrame({"station_id": "5000.01", "ts_utc": hours[:2], "bikes": [3, 1],
                                       "proj_1h": [None, 1.5], "proj_3h": [None, None]})
    api.cache.clear()
    monkeypatch.setattr(api, "fetch", lambda name: STATUS if name == "station_status" else INFO)
    corr = tmp_path / "corr.json"
    corr.write_text(json.dumps({t: {str(h): 1.0 for h in range(1, 7)} for t in ("pickups", "dropoffs")}))
    monkeypatch.setattr(rebalance, "P90_SUM", str(corr))
    monkeypatch.setattr(rebalance, "THRESHOLDS", str(tmp_path / "missing.json"))  # default levels
    return TestClient(api.app)  # not as a context manager: no background refresh thread


def test_forecast(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    r = c.get("/forecast/5000.01").json()
    assert len(r["hours"]) == 6 and r["hours"][0]["pu_mean"] == 2
    assert c.get("/forecast/nope").status_code == 404


def test_alerts_and_moves(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    a = c.get("/alerts").json()
    assert a["renting_stations"] == 3
    flagged = {s["station_id"]: s for s in a["stations"]}
    assert flagged.keys() == {"5000.01", "5000.03"}
    assert flagged["5000.01"]["empty"] and flagged["5000.03"]["full"]
    assert [s["station_id"] for s in c.get("/alerts?kind=full").json()["stations"]] == ["5000.03"]
    moves = c.get("/recommendations").json()["moves"]
    assert {(m["from_id"], m["to_id"], m["reason"]) for m in moves} <= {("5000.02", "5000.01", "empty"),
                                                                       ("5000.03", "5000.02", "full")}
    assert moves


def test_accuracy_and_track(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    assert c.get("/accuracy").json()["summary"] == []
    t = c.get("/track/5000.01").json()
    assert [h["bikes"] for h in t] == [3, 1] and t[1]["proj_1h"] == 1.5
    assert c.get("/track/nope").status_code == 404


def test_track_aligns_projections_with_actuals():
    from src.api.publish import track  # also checks publish (and backtest's imports) still import

    t0 = pd.Timestamp("2026-10-20 10:00", tz="UTC")
    a = pd.DataFrame({"station_id": "s", "t0": pd.date_range(t0, periods=4, freq="h"), "bikes": [10, 8, 6, 4],
                      **{f"bikes_{h}h": [10 - 2 * h - 0.5] * 4 for h in range(1, 7)}})
    t = track(a, t0).set_index("ts_utc")
    # projection for 11:00 made at 10:00 (1 h) = 7.5; for 13:00 made at 10:00 (3 h) = 3.5
    assert t.loc[t0 + pd.Timedelta(hours=1), "proj_1h"] == 7.5
    assert t.loc[t0 + pd.Timedelta(hours=3), "proj_3h"] == 3.5
    assert pd.isna(t.loc[t0, "proj_1h"])


def test_combine_weather_prefers_archive_for_past_and_live_for_future(tmp_path):
    from src.api.publish import combine_weather

    def csv(name, temp, hours):
        pd.DataFrame({"time": [f"2026-10-26T{h}:00" for h in hours], "temp_c": temp}).to_csv(tmp_path / name, index=False)

    csv("h.csv", 1, [10, 12, 13])
    csv("l.csv", 2, [10, 11, 12, 13])
    out = combine_weather(str(tmp_path / "h.csv"), str(tmp_path / "l.csv"), pd.Timestamp("2026-10-26 12:00", tz="UTC"),
                          tmp_path / "w.csv")
    # 10:00 archive; 11:00 missing from the archive -> live; 12:00 and later -> live forecast
    assert pd.read_csv(out)["temp_c"].tolist() == [1, 2, 2, 2]


def test_warming_up(monkeypatch):
    api.state["fc"] = None
    c = TestClient(api.app)
    assert c.get("/alerts").status_code == 503
