"""Data-quality checks on the built Phase 2 tables. Run `python -m src.ingest.station_hour` first."""
import os

import pytest

from src.ingest.station_hour import P, TZ, connect

pytestmark = pytest.mark.skipif(
    not os.path.exists(f"{P}station_hour.parquet"), reason="build data first: python -m src.ingest.station_hour"
)
con = connect()
SH, TRIPS, ST = (f"'{P}{n}.parquet'" for n in ("station_hour", "trips_clean", "stations"))


def one(sql):
    return con.sql(sql).fetchone()


def test_no_negative_counts():
    assert one(f"SELECT min(pickups) >= 0 AND min(dropoffs) >= 0 FROM {SH}")[0]


def test_no_duplicate_station_hours():
    n, distinct = one(f"SELECT count(*), count(DISTINCT (station_id, ts_utc)) FROM {SH}")
    assert n == distinct


def test_grid_has_no_holes():
    # Every station has one row for every hour between its first and last hour.
    holes = one(f"""
        SELECT count(*) FROM (
            SELECT station_id FROM {SH} GROUP BY station_id
            HAVING count(*) <> date_diff('hour', min(ts_utc), max(ts_utc)) + 1)
    """)[0]
    assert holes == 0


def test_daily_totals_match_trips():
    # Catches trips lost by the grid or duplicated by joins. A day can have zero trips: the system
    # shut down for a snowstorm from 2026-02-22 20:00 to 2026-02-24 05:00.
    bad = one(f"""
        WITH t AS (
            SELECT timezone('{TZ}', started_utc)::DATE AS d, count(*) AS n FROM {TRIPS} GROUP BY 1),
        g AS (SELECT ts_local::DATE AS d, sum(pickups) AS n FROM {SH} GROUP BY 1)
        SELECT count(*) FROM t FULL JOIN g USING (d) WHERE coalesce(t.n, 0) <> coalesce(g.n, 0)
    """)[0]
    assert bad == 0
    pu, do = one(f"SELECT sum(pickups), sum(dropoffs) FROM {SH}")
    trips, ended_in_window = one(f"""
        SELECT count(*), count(*) FILTER (WHERE ended_utc < (SELECT max(ts_utc) FROM {SH}) + INTERVAL 1 HOUR)
        FROM {TRIPS}
    """)
    assert pu == trips and do == ended_in_window


def test_weather_has_no_gaps():
    assert one(f"SELECT count(*) FROM {SH} WHERE temp_c IS NULL OR precip_mm IS NULL")[0] == 0


def test_dst_days_have_right_hour_count():
    hours = dict(con.sql(f"""
        SELECT ts_local::DATE::VARCHAR, count(DISTINCT ts_utc) FROM {SH}
        WHERE ts_local::DATE IN ('2025-11-02', '2026-03-08', '2026-06-01') GROUP BY 1
    """).fetchall())
    assert hours == {"2025-11-02": 25, "2026-03-08": 23, "2026-06-01": 24}


def test_station_ids_are_canonical_and_unique():
    n, distinct, split = one(rf"""
        SELECT count(*), count(DISTINCT station_id),
               count(*) FILTER (WHERE regexp_matches(station_id, '^\d+\.\d$'))
        FROM {ST}
    """)
    assert n == distinct and split == 0
