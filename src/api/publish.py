"""Build what the API serves: next-48 h forecasts + a 7-day track record against real GBFS stock.

    python -m src.api.publish            # writes data/serving/ (the serve workflow uploads it to Hugging Face)

Runs every 3 h in .github/workflows/serve.yml, so the API host needs no models or trip history.
- forecasts.parquet: target hours [now, now + 48 h), weather from the live Open-Meteo forecast.
- Track record: forecasts for the past 7 days rebuilt with *archived* forecast weather (what serving would have
  seen; forecasts depend on the target hour + weather only), scored like src.decision.backtest against the GBFS
  snapshots committed by the collector. Uses the frozen tuned alert levels; nothing is fitted here.
  scorecard.json: precision/recall per kind x variant (7 days and per local day); track.parquet: actual bikes vs
  the bikes projected 1 h and 3 h earlier, last 48 h per station. The newest 3 h aren't scored yet (window open).
"""
import json
import shutil
from pathlib import Path

import pandas as pd

from src.decision.backtest import pairs, prf
from src.decision.rebalance import LOOKAHEAD_H, THRESHOLDS, assess, levels, risk_scores
from src.features.build import P, TZ, connect
from src.ingest import weather
from src.ingest.station_hour import build_gbfs_status
from src.models.predict import P90_SUM, forecast_range

OUT = Path("data/serving")
WINDOW_H = 48
TRACK_DAYS = 7
NOTE = ("Alerts flag a station when the projected stock reaches the alert level within 3 h. 'Actual' = the station "
        "really hit 0 bikes (empty) or 0 free docks (full) in a GBFS snapshot within those 3 h. Citi Bike's own "
        "rebalancing averts some stockouts, so precision is a lower bound. 'Caught early' = recall on stockouts at "
        "stations that were not low when the alert was made (the 'current stock' baseline catches none of these).")


def scorecard(a: pd.DataFrame, lv: dict, persistence: dict) -> dict:
    sc = risk_scores(a)
    day = a["t0"].dt.tz_convert(TZ).dt.date.astype(str)
    summary, daily = [], []
    for kind in ("empty", "full"):
        actual = a[f"actual_{kind}"]
        new = sc[f"{kind}_persistence"] > 2  # not low now (default level)
        for variant, col, level in (("point", kind, lv[kind]), ("cautious", f"{kind}_cautious", lv[f"{kind}_cautious"]),
                                    ("current stock", f"{kind}_persistence", persistence[f"{kind}_persistence"])):
            alert = sc[col] <= level
            summary.append({"kind": kind, "variant": variant, **prf(alert, actual),
                            "caught_early": prf(alert[new], actual[new])["recall"]})
            for d, m in day.groupby(day).groups.items():
                daily.append({"date": d, "kind": kind, "variant": variant, **prf(alert[m], actual[m])})
    return {"summary": summary, "daily": daily, "note": NOTE}


def track(a: pd.DataFrame, since: pd.Timestamp) -> pd.DataFrame:
    """Per station x hour: actual bikes at the hour vs the bikes projected for it 1 h and 3 h before."""
    act = a.loc[a.t0 >= since, ["station_id", "t0", "bikes"]].rename(columns={"t0": "ts_utc"})
    for h in (1, LOOKAHEAD_H):
        p = a[["station_id", "t0", f"bikes_{h}h"]].rename(columns={f"bikes_{h}h": f"proj_{h}h"})
        p["ts_utc"] = p.pop("t0") + pd.Timedelta(hours=h)
        act = act.merge(p, on=["station_id", "ts_utc"], how="left")
    return act.round({"proj_1h": 2, f"proj_{LOOKAHEAD_H}h": 2}).sort_values(["station_id", "ts_utc"], ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").floor("h")
    start = now - pd.Timedelta(days=TRACK_DAYS)
    naive = lambda t: str(t.tz_localize(None))  # noqa: E731  forecast_range takes naive UTC strings

    live = weather.main("", "", api="forecast").as_posix()
    fc = forecast_range(naive(now), naive(now + pd.Timedelta(hours=WINDOW_H)), live)
    fc.to_parquet(OUT / "forecasts.parquet")

    hist = weather.main(str((start - pd.Timedelta(days=1)).date()), str((now + pd.Timedelta(days=1)).date()),
                        api="historical-forecast", out=OUT / "weather_track.csv").as_posix()
    past = pd.concat([forecast_range(naive(start), naive(now), hist), fc], ignore_index=True)

    status = f"{P}gbfs_status_track.parquet"
    build_gbfs_status(connect(), "data/raw/gbfs/status/*/*.csv.gz", status)
    p = pairs(status)
    p = p.assign(t0=p.t0.dt.tz_convert("UTC"))  # DuckDB returns the machine's timezone
    p = p[p.t0 >= start].reset_index(drop=True)
    with open(P90_SUM) as f:
        a = assess(p, past, json.load(f))
    with open(THRESHOLDS) as f:
        persistence = json.load(f)["persistence"]
    card = {"built_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "forecast_window_utc": [str(fc.ts_utc.min()), str(fc.ts_utc.max())],
            "scored_window_utc": [str(a.t0.min()), str(a.t0.max())], "origins": int(a.t0.nunique()),
            "levels": levels(), **scorecard(a, levels(), persistence)}
    (OUT / "scorecard.json").write_text(json.dumps(card, indent=1))
    track(a, now - pd.Timedelta(hours=WINDOW_H)).to_parquet(OUT / "track.parquet")
    for f in (P90_SUM, THRESHOLDS):  # the API's decision layer reads these two
        shutil.copy(f, OUT)
    print(f"{OUT}: {len(fc):,} forecast rows; scored {card['origins']} origins {card['scored_window_utc']}")
    for r in card["summary"]:
        print(f"  {r['kind']:5} {r['variant']:13} precision {r['precision']:.2f} recall {r['recall']:.2f} "
              f"caught early {r['caught_early']:.2f} ({r['alerts']} alerts, {r['stockouts']} stockouts)")


if __name__ == "__main__":
    main()
