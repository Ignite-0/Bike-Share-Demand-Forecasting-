"""Leakage, availability, masking and shape checks for src/features/build.py on a small synthetic table.

The availability rule is recomputed here in pandas, independently of the SQL, so a bug in the SQL cutoff
can't hide behind itself.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build import MAX_HORIZON_H, PUBLISH_DAY, SHUTDOWN, TIERS, TZ, build, connect

X_LOCAL = pd.Timestamp("2026-04-01 00:00")  # targets at or after this local time get mutated
X_UTC = X_LOCAL.tz_localize(TZ).tz_convert("UTC")


def cutoff(ts_local: pd.Series) -> pd.Series:
    """Local time before which trip data is published, for target hours at ts_local."""
    shifted = ts_local - pd.Timedelta(hours=MAX_HORIZON_H - 1) - pd.Timedelta(days=PUBLISH_DAY)
    return shifted.dt.to_period("M").dt.start_time


def synthetic(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-20 05:00", "2026-07-10 04:00", freq="h", tz="UTC")
    rows = []
    for i in range(5):
        rows.append(pd.DataFrame({
            "station_id": f"{1000 + i}.01", "ts_utc": ts, "ts_local": ts.tz_convert(TZ).tz_localize(None),
            "pickups": rng.poisson(3 + i, len(ts)), "dropoffs": rng.poisson(3 + i, len(ts)),
            "temp_c": rng.normal(10, 8, len(ts)), "feels_c": rng.normal(8, 8, len(ts)),
            "precip_mm": rng.exponential(0.2, len(ts)), "snow_cm": 0.0, "wind_kmh": rng.normal(15, 5, len(ts)),
            "is_holiday": False,
        }))
    return pd.concat(rows, ignore_index=True)


def run(tmp, df, name):
    src, out = f"{tmp}/{name}_sh.parquet", f"{tmp}/{name}_features"
    df.to_parquet(src, index=False)
    stations = f"{tmp}/stations.parquet"
    pd.DataFrame({"station_id": df["station_id"].unique(), "capacity": 30, "lat": 40.7, "lng": -74.0}) \
        .to_parquet(stations, index=False)
    build(connect("2GB"), src=f"'{src}'", out=out, stations=f"'{stations}'", batch=2)  # 3 parts
    return connect("2GB").sql(f"SELECT * FROM '{out}/*.parquet'").df() \
        .sort_values(["station_id", "ts_utc"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("feat").as_posix()
    df = synthetic(0)
    mutated = df.copy()
    mutated.loc[mutated["ts_local"] >= X_LOCAL, ["pickups", "dropoffs"]] = 999
    return df, run(tmp, df, "base"), run(tmp, mutated, "mut")


def masked(src: pd.DataFrame) -> pd.DataFrame:
    shut = (src["ts_local"] >= SHUTDOWN[0]) & (src["ts_local"] <= SHUTDOWN[1])
    return src[~shut]


def test_no_feature_reads_data_it_could_not_have(built):
    _, base, mut = built
    unchanged = {
        "A": pd.Series(True, index=base.index),  # no trip data at all
        "B": base["ts_utc"] < X_UTC + pd.Timedelta(hours=6),  # lags >= 6 h
        "C": cutoff(base["ts_local"]) <= X_LOCAL,  # only months published before the forecast origin
    }
    for tier, keep in unchanged.items():
        cols = TIERS[tier]
        pd.testing.assert_frame_equal(base.loc[keep, cols], mut.loc[keep, cols], obj=f"tier {tier}")
        if not keep.all():  # the test must be able to fail: later rows do see the mutation
            assert not base.loc[~keep, cols].equals(mut.loc[~keep, cols]), f"tier {tier} never changes"


def test_sql_cutoff_matches_rule(built):
    _, base, _ = built
    assert (base["cutoff"] == cutoff(base["ts_local"])).all()


@pytest.mark.parametrize("when", ["2026-06-10 08:00", "2026-07-08 17:00"])  # a train row and a val row
def test_tier_c_values_use_only_published_history(built, when):
    src, base, _ = built
    row = base[(base["station_id"] == "1002.01") & (base["ts_local"] == pd.Timestamp(when))].iloc[0]
    cut = cutoff(pd.Series([row["ts_local"]]))[0]
    h = masked(src[src["station_id"] == "1002.01"])
    before = h[h["ts_local"] < cut]
    same = before[(before["ts_local"].dt.hour == row["hour"]) & (before["ts_local"].dt.dayofweek < 5)]
    recent = before[before["ts_local"] >= cut - pd.Timedelta(days=28)]
    recent_same = same[same["ts_local"] >= cut - pd.Timedelta(days=28)]
    assert row["pu_profile"] == pytest.approx(same["pickups"].mean(), rel=1e-5)
    assert row["pu_recent_profile"] == pytest.approx(recent_same["pickups"].mean(), rel=1e-5)
    assert row["pu_recent_mean7d"] == pytest.approx(
        recent[recent["ts_local"] >= cut - pd.Timedelta(days=7)]["pickups"].mean(), rel=1e-5)


def test_shutdown_is_masked(built):
    src, base, _ = built
    shut_utc = set(src["ts_utc"]) - set(masked(src)["ts_utc"])
    assert not base["ts_utc"].isin(shut_utc).any(), "shutdown hours must not be targets"
    reads_shutdown = (base["ts_utc"] - pd.Timedelta(hours=6)).isin(shut_utc)
    assert reads_shutdown.any() and base.loc[reads_shutdown, "pu_lag6"].isna().all()
    # A 24 h window fully inside the 33 h shutdown has no valid hours -> NULL, not 0.
    inside = base["ts_utc"] == min(shut_utc) + pd.Timedelta(hours=29)
    assert base.loc[inside, "pu_roll24"].isna().all()


def test_shape_and_splits(built):
    _, base, _ = built
    assert not base.duplicated(["station_id", "ts_utc"]).any()
    by = base.groupby("split")["ts_local"]
    assert by.max()["train"] < by.min()["val"]
    assert base[TIERS["A"]].notna().all().all(), "tier A must always be available"


def test_publish_day_covers_observed_release_dates():
    # Citi Bike published the 2026 monthly files on day 4, 4, 3, 4, 4, 6, 12, 10 and 5 of the next month
    # (S3 LastModified). The cutoff must not assume data earlier than the slowest observed release.
    assert PUBLISH_DAY > 12
