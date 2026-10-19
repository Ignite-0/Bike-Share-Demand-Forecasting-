"""Feature table for training AND serving: one row per station per target hour.

Time rules (see reports/EDA_REPORT.md for why each feature exists):
- A forecast issued at origin t0 (a whole UTC hour) knows only completed hours [x, x+1h) with x+1h <= t0.
- The target hour is T = t0 + (h-1) for horizon h = 1..6, so the earliest possible origin is T - 5 h.
- Shutdown hours (snowstorm, Feb 22-24) are never targets, and every trip-based feature reads the *masked*
  targets: a lag landing in the shutdown is NULL, never 0, and means count valid hours only.

Tiers. TIERS only groups column names; the SQL below is what enforces each tier's time rule, and
tests/test_features.py checks it by mutating future targets.
- A, always available live: calendar, daylight, weather (forecast at serving), station static. No trip data.
- C, published trips: trip data is released one month at a time (2026 files appeared on the 3rd-12th of the
  next month), so a month is treated as available from the 15th of the following month. For target hour T,
  tier C reads only local hours before
      cutoff(T) = first day of the month containing (T - 5 h - 15 days)
  through pre-aggregated tables joined on (station, cutoff), never through row windows. The same SQL runs for
  train, val, test and serving, so every split obeys the same availability rule. Profiles therefore only use
  earlier months, never later training months.
- B, fresh trips, lags >= 6 h (UTC hours; valid for every horizon h <= 6). UPPER BOUND ONLY: live, these
  inputs would need a GBFS-based estimate that cannot be evaluated on the historical window. Never report a
  model using B as production performance (PRODUCTION_TIERS).

cluster and volume_tier are reporting labels (fit on train months), not model features.
Usage: python -m src.features.build
"""
import math
import os
from datetime import date, timedelta

import duckdb
import holidays
import numpy as np

TZ = "America/New_York"
LAT = 40.75
P = "data/processed/"
SHUTDOWN = ("2026-02-22 20:00", "2026-02-24 05:00")  # local, from the EDA
STORM = ("2026-09-26", "2026-09-28")  # local dates, test month
SPLITS = ("2026-07-01", "2026-09-01")  # local: train < first <= val < second <= test
TRAIN_END = SPLITS[0]
PUBLISH_DAY = 15  # a month's trip file counts as available from this day of the next month
MAX_HORIZON_H = 6
MIN_PROFILE_OBS = 4

TIERS = {
    "A": ["hour", "dow", "is_weekend", "is_holiday", "days_to_holiday", "days_from_holiday", "daylight_h",
          "doy_sin", "doy_cos", "temp_c", "feels_c", "precip_mm", "is_rain", "snow_cm", "wind_kmh",
          "capacity", "lat", "lng"],
    "B": [f"{t}_lag{k}" for t in ("pu", "dr") for k in (6, 24, 168, 336)]
         + [f"{t}_{n}" for t in ("pu", "dr") for n in ("same_hour_4w", "roll24", "roll7d")],
    "C": [f"{t}_{n}" for t in ("pu", "dr") for n in ("profile", "recent_profile", "recent_mean7d")],
}
PRODUCTION_TIERS = ("A", "C")
TARGETS = ["pickups", "dropoffs"]
REPORT = ["cluster", "volume_tier", "is_storm", "cutoff"]

# Local timestamp before which trip data is published, for a target hour at local time `ts`.
CUTOFF_SQL = (f"date_trunc('month', {{ts}} - INTERVAL {MAX_HORIZON_H - 1} HOUR"
              f" - INTERVAL {PUBLISH_DAY} DAY)::TIMESTAMP")


def connect(memory_limit: str = "8GB") -> duckdb.DuckDBPyConnection:
    os.makedirs("data/duckdb_tmp", exist_ok=True)
    con = duckdb.connect()
    con.sql(f"SET TimeZone = 'UTC'; SET enable_progress_bar = false; SET memory_limit = '{memory_limit}';"
            "SET preserve_insertion_order = false; SET temp_directory = 'data/duckdb_tmp'")
    return con


def calendar_table(con, d0: date, d1: date) -> None:
    """Per local date: distance to holidays, daylight length, day-of-year cycle."""
    hol = sorted(holidays.country_holidays("US", subdiv="NY", years=range(d0.year - 1, d1.year + 2)))
    rows, d = [], d0
    while d <= d1:
        nxt = next(h for h in hol if h >= d)
        prv = next(h for h in reversed(hol) if h <= d)
        doy = d.timetuple().tm_yday
        decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365)
        daylight = 24 / math.pi * math.acos(-math.tan(math.radians(LAT)) * math.tan(decl))
        a = 2 * math.pi * (doy - 1) / 365.25
        rows.append((d, min((nxt - d).days, 30), min((d - prv).days, 30), daylight, math.sin(a), math.cos(a)))
        d += timedelta(days=1)
    con.sql("CREATE OR REPLACE TEMP TABLE cal (d DATE, days_to_holiday INT, days_from_holiday INT, "
            "daylight_h DOUBLE, doy_sin DOUBLE, doy_cos DOUBLE)")
    con.executemany("INSERT INTO cal VALUES (?, ?, ?, ?, ?, ?)", rows)


def _mean(exprs: list[str]) -> str:
    """Mean of the non-NULL expressions, NULL if all are NULL."""
    n = " + ".join(f"({e} IS NOT NULL)::INT" for e in exprs)
    s = " + ".join(f"coalesce({e}, 0)" for e in exprs)
    return f"(({s}) / nullif({n}, 0))"


def _roll(col: str, first: int, last: int) -> str:
    """Mean of col over hours [T-first, T-last], NULL when fewer than half the hours are valid."""
    frame = f"(st RANGE BETWEEN INTERVAL {first} HOUR PRECEDING AND INTERVAL {last} HOUR PRECEDING)"
    need = (first - last + 1) / 2
    return f"CASE WHEN count({col}) OVER {frame} >= {need} THEN avg({col}) OVER {frame} END"


def feature_sql(src: str, stations: str, clusters: str) -> str:
    """SELECT producing the feature table from a station_hour-shaped relation `src`.

    Serving passes history + future hours (future targets NULL, weather from the forecast) through the same SQL.
    Tier B uses LAG() over each station's hourly grid, so `src` must be contiguous per station.
    """
    b = []
    for t in ("pu", "dr"):
        b += [f"lag({t}, {k}) OVER st AS {t}_lag{k}" for k in (6, 24, 168, 336)]
        b.append(f"{_mean([f'lag({t}, {k}) OVER st' for k in (168, 336, 504, 672)])} AS {t}_same_hour_4w")
        b.append(f"{_roll(t, 29, 6)} AS {t}_roll24")
        b.append(f"{_roll(t, 173, 6)} AS {t}_roll7d")
    f32 = lambda cols: ", ".join(f"{x}::FLOAT AS {x}" for x in cols)  # noqa: E731
    return f"""
        WITH s AS (
            SELECT h.station_id, h.ts_utc, h.ts_local, h.pickups, h.dropoffs,
                   h.temp_c, h.feels_c, h.precip_mm, h.snow_cm, h.wind_kmh, h.is_holiday,
                   h.ts_local BETWEEN '{SHUTDOWN[0]}' AND '{SHUTDOWN[1]}' AS shutdown,
                   hour(h.ts_local) AS hour, dayofweek(h.ts_local) AS dow,
                   dayofweek(h.ts_local) IN (0, 6) AS is_weekend,
                   CASE WHEN dayofweek(h.ts_local) IN (0, 6) THEN 1 WHEN h.is_holiday THEN 2 ELSE 0 END AS daytype,
                   date_trunc('month', h.ts_local)::TIMESTAMP AS month,
                   {CUTOFF_SQL.format(ts='h.ts_local')} AS cutoff
            FROM {src} h
        ),
        m AS (  -- masked targets: shutdown hours become NULL so they never feed a lag or mean
            SELECT *, CASE WHEN NOT shutdown THEN pickups END AS pu,
                      CASE WHEN NOT shutdown THEN dropoffs END AS dr
            FROM s
        ),

        -- Tier C. Each aggregate is keyed by the cutoff it is valid for, and built only from hours < cutoff.
        mon AS (
            SELECT station_id, hour, daytype, month,
                   sum(pu) AS spu, count(pu) AS npu, sum(dr) AS sdr, count(dr) AS ndr
            FROM m GROUP BY ALL
        ),
        prof AS (  -- running totals through `month`, usable from cutoff = next month
            SELECT station_id, hour, daytype, month + INTERVAL 1 MONTH AS usable_from,
                   sum(spu) OVER c AS spu, sum(npu) OVER c AS npu, sum(sdr) OVER c AS sdr, sum(ndr) OVER c AS ndr
            FROM mon
            WINDOW c AS (PARTITION BY station_id, hour, daytype ORDER BY month ROWS UNBOUNDED PRECEDING)
        ),
        cuts AS (SELECT DISTINCT station_id, cutoff FROM m),
        recent AS (  -- the 28 days before each cutoff
            SELECT k.station_id, k.cutoff, m.hour, m.daytype, m.ts_local, m.pu, m.dr
            FROM cuts k JOIN m ON m.station_id = k.station_id
             AND m.ts_local >= k.cutoff - INTERVAL 28 DAY AND m.ts_local < k.cutoff
        ),
        rprof AS (
            SELECT station_id, cutoff, hour, daytype,
                   CASE WHEN count(pu) >= 2 THEN avg(pu) END AS pu_recent_profile,
                   CASE WHEN count(dr) >= 2 THEN avg(dr) END AS dr_recent_profile
            FROM recent GROUP BY ALL
        ),
        r7 AS (
            SELECT station_id, cutoff,
                   CASE WHEN count(pu) >= 84 THEN avg(pu) END AS pu_recent_mean7d,
                   CASE WHEN count(dr) >= 84 THEN avg(dr) END AS dr_recent_mean7d
            FROM recent WHERE ts_local >= cutoff - INTERVAL 7 DAY GROUP BY ALL
        ),

        w AS (  -- Tier B: row windows over the station's hourly grid
            SELECT m.*, {", ".join(b)}
            FROM m
            WINDOW st AS (PARTITION BY station_id ORDER BY ts_utc)
        )
        SELECT w.station_id, w.ts_utc, w.ts_local,
               CASE WHEN w.ts_local < '{SPLITS[0]}' THEN 'train'
                    WHEN w.ts_local < '{SPLITS[1]}' THEN 'val' ELSE 'test' END AS split,
               w.pickups::SMALLINT AS pickups, w.dropoffs::SMALLINT AS dropoffs,
               w.hour::TINYINT AS hour, w.dow::TINYINT AS dow, w.is_weekend, w.is_holiday,
               cal.days_to_holiday::TINYINT AS days_to_holiday, cal.days_from_holiday::TINYINT AS days_from_holiday,
               {f32(["daylight_h", "doy_sin", "doy_cos"])},
               {f32(["temp_c", "feels_c", "precip_mm", "snow_cm", "wind_kmh"])}, w.precip_mm >= 0.1 AS is_rain,
               sta.capacity::SMALLINT AS capacity, sta.lat::FLOAT AS lat, sta.lng::FLOAT AS lng,
               {f32(TIERS["B"])},
               CASE WHEN prof.npu >= {MIN_PROFILE_OBS} THEN prof.spu / prof.npu END::FLOAT AS pu_profile,
               CASE WHEN prof.ndr >= {MIN_PROFILE_OBS} THEN prof.sdr / prof.ndr END::FLOAT AS dr_profile,
               rprof.pu_recent_profile::FLOAT AS pu_recent_profile, rprof.dr_recent_profile::FLOAT AS dr_recent_profile,
               r7.pu_recent_mean7d::FLOAT AS pu_recent_mean7d, r7.dr_recent_mean7d::FLOAT AS dr_recent_mean7d,
               cl.cluster::TINYINT AS cluster, cl.volume_tier,
               w.ts_local::DATE BETWEEN '{STORM[0]}' AND '{STORM[1]}' AS is_storm, w.cutoff
        FROM w
        JOIN cal ON cal.d = w.ts_local::DATE
        ASOF LEFT JOIN prof ON prof.station_id = w.station_id AND prof.hour = w.hour
                           AND prof.daytype = w.daytype AND w.cutoff >= prof.usable_from
        LEFT JOIN rprof ON rprof.station_id = w.station_id AND rprof.cutoff = w.cutoff
                       AND rprof.hour = w.hour AND rprof.daytype = w.daytype
        LEFT JOIN r7 ON r7.station_id = w.station_id AND r7.cutoff = w.cutoff
        LEFT JOIN {stations} sta ON sta.station_id = w.station_id
        LEFT JOIN {clusters} cl ON cl.station_id = w.station_id
        WHERE NOT w.shutdown
    """


def fit_clusters(con, src: str, out: str, k: int = 4) -> None:
    """Reporting labels: k-means on each station's weekday 24 h shape, train months only.

    Clusters are named by morning net flow (same rule as the EDA) so names survive a rerun.
    """
    from sklearn.cluster import KMeans

    prof = con.sql(f"""
        SELECT station_id, hour(ts_local) AS h, avg(pickups) AS pu, avg(dropoffs) AS dr
        FROM {src}
        WHERE ts_local < '{TRAIN_END}' AND dayofweek(ts_local) NOT IN (0, 6) AND NOT is_holiday
          AND NOT ts_local BETWEEN '{SHUTDOWN[0]}' AND '{SHUTDOWN[1]}'
        GROUP BY ALL
    """).df().pivot(index="station_id", columns="h", values=["pu", "dr"]).fillna(0)
    daily = prof.sum(axis=1)  # weekday pick-ups + drop-offs
    x = prof[daily >= 5]
    x = x.div(x.sum(axis=1), axis=0)
    km = KMeans(min(k, len(x)), n_init=10, random_state=0).fit(x)
    am = km.cluster_centers_[:, 24 + 7:24 + 10].sum(1) - km.cluster_centers_[:, 7:10].sum(1)  # dr - pu, 7-9h
    rank = {old: new for new, old in enumerate(np.argsort(am))}  # 0 = residential ... k-1 = business
    labels = prof.index.to_series().map(dict(zip(x.index, (rank[c] for c in km.labels_)))).fillna(-1)
    tiers = np.select([daily < 5, daily < 20, daily < 80], ["<5/day", "5-20/day", "20-80/day"], "80+/day")
    con.register("_cl", labels.rename("cluster").to_frame().assign(volume_tier=tiers).reset_index())
    con.sql(f"COPY (SELECT station_id, cluster::INT AS cluster, volume_tier FROM _cl) TO '{out}' (FORMAT parquet)")
    con.unregister("_cl")


def build(con, src: str = f"'{P}station_hour.parquet'", out: str = f"{P}features",
          stations: str = f"'{P}stations.parquet'", batch: int = 500) -> None:
    """Write the feature table to `out`/part-NN.parquet, one part per batch of stations.

    Every feature is computed within one station, so batching changes nothing but peak memory
    (measured ~0.9 GB per 100 stations; 500 per batch stays well under the 8 GB DuckDB limit).
    Read it back as read_parquet('<out>/*.parquet'). Paths must use forward slashes (pasted into SQL).
    """
    import shutil

    d0, d1 = con.sql(f"SELECT min(ts_local)::DATE, max(ts_local)::DATE FROM {src}").fetchone()
    calendar_table(con, d0, d1)
    labels = out.rsplit("/", 1)[0] + "/station_labels.parquet"
    fit_clusters(con, src, labels)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    ids = [r[0] for r in con.sql(f"SELECT DISTINCT station_id FROM {src} ORDER BY 1").fetchall()]
    for i in range(0, len(ids), batch):
        con.sql("CREATE OR REPLACE TEMP TABLE batch (station_id VARCHAR)")
        con.executemany("INSERT INTO batch VALUES (?)", [(x,) for x in ids[i:i + batch]])
        part = f"(SELECT * FROM {src} WHERE station_id IN (SELECT station_id FROM batch))"
        sql = feature_sql(part, stations, f"'{labels}'")
        con.sql(f"COPY ({sql}) TO '{out}/part-{i // batch:02d}.parquet' (FORMAT parquet, ROW_GROUP_SIZE 500000)")


if __name__ == "__main__":
    import time

    t = time.time()
    con = connect()
    build(con)
    n = con.sql(f"SELECT split, count(*) FROM '{P}features/*.parquet' GROUP BY 1 ORDER BY 1").fetchall()
    print(f"{P}features/: {n} in {time.time() - t:.0f}s")
