"""Build the modeling table: one row per station per hour, with pickups, drop-offs, weather, holidays.

Steps (each overwrites its output, so rerunning is safe):
  1. trips CSVs   -> data/processed/trips_clean.parquet  (cleaned, timestamps in UTC)
  2. stations     -> data/processed/stations.parquet     (one row per station ID, capacity, is_new)
  3. station-hour -> data/processed/station_hour.parquet (complete grid, zero-filled)
  4. GBFS status  -> data/processed/gbfs_status.parquet  (live snapshots, feed outages marked)

Time: every hour is keyed by ts_utc. Trip CSVs are naive New York local time and are converted here,
the only place in the project that does it. ts_local is kept for calendar features.
Usage: python -m src.ingest.station_hour
"""
import glob
import os

import duckdb
import holidays

TZ = "America/New_York"
START, END = "2025-10-01", "2026-10-01"  # local dates, END exclusive
TRIPS = "data/raw/trips/*/*.csv"
P = "data/processed/"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    # date_trunc on TIMESTAMPTZ uses the session zone; UTC keeps hour buckets real 60-minute hours.
    con.sql("SET TimeZone = 'UTC'; SET enable_progress_bar = false")
    return con


def norm_id(col: str) -> str:
    """SQL for a canonical station ID. Some monthly files drop a trailing zero ("5997.1" vs "5997.10")."""
    return rf"(CASE WHEN regexp_matches(trim({col}), '^\d+\.\d$') THEN trim({col}) || '0' ELSE trim({col}) END)"


def is_public_station(id_col: str, lat_col: str) -> str:
    """SQL predicate. Public NYC stations have IDs like 5997.10; this drops depots and shops (SYS016,
    "Shop Morgan", 1234.56 = Morgan HCT Charging), demo/lab docks, and Jersey City/Hoboken (JC/HB,
    which appear only via cross-river trips in the NYC files, so their demand would look tiny)."""
    return (rf"(regexp_matches({id_col}, '^\d+\.\d+') AND {id_col} <> '1234.56'"
            f" AND {lat_col} BETWEEN 40.4 AND 41.0)")


def clean_trips(con) -> None:
    # ponytail: the fall-back hour (01:00-02:00 local, once a year) is ambiguous in the CSVs; DuckDB
    # resolves it to standard time, so both real hours land in one UTC hour and the other gets 0.
    con.sql(f"""
        COPY (
            WITH t AS (
                SELECT ride_id, rideable_type, member_casual,
                       {norm_id('start_station_id')} AS start_station_id,
                       {norm_id('end_station_id')} AS end_station_id,
                       start_station_name, end_station_name,
                       start_lat::DOUBLE AS start_lat, start_lng::DOUBLE AS start_lng,
                       end_lat::DOUBLE AS end_lat, end_lng::DOUBLE AS end_lng,
                       started_at::TIMESTAMP AS started_local, ended_at::TIMESTAMP AS ended_local
                FROM read_csv('{TRIPS}', all_varchar = true, union_by_name = true)
            )
            SELECT * EXCLUDE (started_local, ended_local),
                   timezone('{TZ}', started_local) AS started_utc,
                   timezone('{TZ}', ended_local) AS ended_utc
            FROM t
            WHERE start_station_id IS NOT NULL AND end_station_id IS NOT NULL  -- dockless parking
              AND {is_public_station('start_station_id', 'start_lat')}
              AND {is_public_station('end_station_id', 'end_lat')}
              AND started_local >= '{START}' AND started_local < '{END}'
              AND ended_local - started_local
                  BETWEEN INTERVAL 60 SECOND AND INTERVAL 24 HOUR  -- false starts, lost bikes
            QUALIFY row_number() OVER (PARTITION BY ride_id) = 1
        ) TO '{P}trips_clean.parquet' (FORMAT parquet)
    """)


def build_stations(con) -> None:
    info = sorted(glob.glob("data/raw/gbfs/info/*.json"))[-1]  # latest metadata snapshot
    con.sql(f"""
        COPY (
            WITH ev AS (
                SELECT start_station_id AS id, start_station_name AS name, start_lat AS lat,
                       start_lng AS lng, started_utc AS ts FROM '{P}trips_clean.parquet'
                UNION ALL
                SELECT end_station_id, end_station_name, end_lat, end_lng, ended_utc
                FROM '{P}trips_clean.parquet'
            ),
            s AS (
                SELECT id AS station_id, mode(name) AS name, median(lat) AS lat, median(lng) AS lng,
                       min(ts) AS first_seen, max(ts) AS last_seen
                FROM ev GROUP BY id
            ),
            info AS (SELECT {norm_id('short_name')} AS short_name, capacity, lat, lon FROM read_json('{info}')),
            -- Stations retired or moved before the metadata snapshot have no ID match;
            -- borrow capacity from the nearest live station within 150 m.
            near AS (
                SELECT s.station_id, arg_min(i.capacity, d) AS capacity
                FROM s, info i,
                     LATERAL (SELECT 111320 * sqrt((s.lat - i.lat) ** 2
                                     + ((s.lng - i.lon) * cos(radians(40.73))) ** 2) AS d)
                WHERE d < 150
                GROUP BY s.station_id
            )
            SELECT s.*, coalesce(i.capacity, near.capacity) AS capacity,
                   i.short_name IS NOT NULL AS id_matched,
                   s.last_seen - s.first_seen < INTERVAL 30 DAY AS is_new
            FROM s
            LEFT JOIN info i ON i.short_name = s.station_id
            LEFT JOIN near USING (station_id)
            ORDER BY station_id
        ) TO '{P}stations.parquet' (FORMAT parquet)
    """)


def build_station_hour(con) -> None:
    hol = holidays.country_holidays("US", subdiv="NY", years=[2025, 2026])
    con.sql("CREATE OR REPLACE TEMP TABLE hol (d DATE, holiday VARCHAR)")
    con.executemany("INSERT INTO hol VALUES (?, ?)", list(hol.items()))
    con.sql(f"""
        COPY (
            WITH win AS (
                SELECT timezone('{TZ}', '{START}'::TIMESTAMP) AS t0,
                       timezone('{TZ}', '{END}'::TIMESTAMP) - INTERVAL 1 HOUR AS t1
            ),
            -- Grid spans each station's own active window: zeros before a station opened are fake.
            grid AS (
                SELECT station_id,
                       unnest(generate_series(greatest(date_trunc('hour', first_seen), t0),
                                              least(date_trunc('hour', last_seen), t1),
                                              INTERVAL 1 HOUR)) AS ts_utc
                FROM '{P}stations.parquet', win
            ),
            pu AS (SELECT start_station_id AS station_id, date_trunc('hour', started_utc) AS ts_utc,
                          count(*) AS n FROM '{P}trips_clean.parquet' GROUP BY ALL),
            dr AS (SELECT end_station_id AS station_id, date_trunc('hour', ended_utc) AS ts_utc,
                          count(*) AS n FROM '{P}trips_clean.parquet' GROUP BY ALL),
            w AS (
                SELECT * REPLACE (timezone('UTC', strptime(ts_utc, '%Y-%m-%dT%H:%M')) AS ts_utc)
                FROM read_csv('data/raw/weather/weather_hourly.csv', header = true, all_varchar = true,
                              names = ['ts_utc', 'temp_c', 'feels_c', 'precip_mm', 'rain_mm',
                                       'snow_cm', 'wind_kmh', 'weather_code'])
            )
            SELECT g.station_id, g.ts_utc, timezone('{TZ}', g.ts_utc) AS ts_local,
                   coalesce(pu.n, 0)::INT AS pickups, coalesce(dr.n, 0)::INT AS dropoffs,
                   (coalesce(dr.n, 0) - coalesce(pu.n, 0))::INT AS net_flow,
                   COLUMNS(w.* EXCLUDE (ts_utc, weather_code))::DOUBLE, w.weather_code::INT AS weather_code,
                   hol.d IS NOT NULL AS is_holiday
            FROM grid g
            LEFT JOIN pu USING (station_id, ts_utc)
            LEFT JOIN dr USING (station_id, ts_utc)
            LEFT JOIN w USING (ts_utc)
            LEFT JOIN hol ON hol.d = timezone('{TZ}', g.ts_utc)::DATE
            ORDER BY g.station_id, g.ts_utc
        ) TO '{P}station_hour.parquet' (FORMAT parquet)
    """)


def build_gbfs_status(con) -> None:
    con.sql(f"""
        COPY (
            WITH s AS (
                SELECT * REPLACE (fetched_at::TIMESTAMPTZ AS fetched_at)
                FROM read_csv('data/raw/gbfs/status/*/*.csv.gz', all_varchar = true)
            ),
            snaps AS (
                SELECT fetched_at,
                       date_diff('minute', lag(fetched_at) OVER (ORDER BY fetched_at), fetched_at) AS gap_min
                FROM (SELECT DISTINCT fetched_at FROM s)
            )
            SELECT s.fetched_at, s.station_id,
                   TRY_CAST(COLUMNS(c -> c NOT IN ('fetched_at', 'station_id', 'last_reported')) AS INT),
                   to_timestamp(s.last_reported::BIGINT) AS last_reported,
                   snaps.gap_min,
                   coalesce(snaps.gap_min > 30, false) AS after_outage  -- feed gap; never interpolate
            FROM s JOIN snaps USING (fetched_at)
            ORDER BY s.fetched_at, s.station_id
        ) TO '{P}gbfs_status.parquet' (FORMAT parquet)
    """)


def main() -> None:
    os.makedirs(P, exist_ok=True)
    con = connect()
    for step in (clean_trips, build_stations, build_station_hour, build_gbfs_status):
        step(con)
        print(f"{step.__name__}: done")
    for name in ("trips_clean", "stations", "station_hour", "gbfs_status"):
        n = con.sql(f"SELECT count(*) FROM '{P}{name}.parquet'").fetchone()[0]
        print(f"{P}{name}.parquet: {n:,} rows")


if __name__ == "__main__":
    main()
