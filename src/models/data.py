"""Read the Phase 4 feature table. The test split is guarded: only the final run may load it."""
import duckdb
import pandas as pd

from src.features.build import P, TIERS

FEATURES = f"'{P}features/*.parquet'"
STOP_FROM = "2026-06-01"  # early-stopping set = June, the last training month (never val)


SEASON = ["doy_sin", "doy_cos", "daylight_h"]


def feature_cols(feature_set: str) -> list[str]:
    """'A+C+sid' -> tier A and C columns plus station_id; a '-season' suffix drops SEASON.
    Built from TIERS, never hand-copied."""
    base, _, drop = feature_set.partition("-")
    if drop not in ("", "season"):
        raise ValueError(f"unknown drop '{drop}' in {feature_set}")
    parts = base.split("+")
    cols = [c for p in parts if p != "sid" for c in TIERS[p]] + (["station_id"] if "sid" in parts else [])
    return [c for c in cols if not (drop == "season" and c in SEASON)]


def load(split: str, columns: list[str], final: bool = False, where: str = "") -> pd.DataFrame:
    """split: one split name or several joined by ',' (e.g. 'train,val')."""
    splits = split.split(",")
    if "test" in splits and not final:
        raise PermissionError("test split is reserved for the single final run (final=True)")
    cols = ", ".join(dict.fromkeys(columns))
    inlist = ", ".join(f"'{s}'" for s in splits)
    df = duckdb.sql(f"SELECT {cols} FROM {FEATURES} WHERE split IN ({inlist}) {where}").df()
    if "station_id" in df:
        df["station_id"] = df["station_id"].astype("category")
    return df
