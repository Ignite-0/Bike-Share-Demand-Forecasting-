import io
import zipfile

from src.ingest.download_trips import MONTHLY, extract_csvs


def test_monthly_pattern():
    assert MONTHLY.match("202510-citibike-tripdata.zip")
    assert MONTHLY.match("202301-citibike-tripdata.csv.zip")
    assert not MONTHLY.match("JC-202510-citibike-tripdata.csv.zip")
    assert not MONTHLY.match("2022-citibike-tripdata.zip")


def test_extract_flattens_nested_zips(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("b.csv", "x\n")
    outer = tmp_path / "m.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("folder/a.csv", "x\n")
        z.writestr("folder/inner.zip", inner.getvalue())
        z.writestr("__MACOSX/folder/._a.csv", "junk")
    dest = tmp_path / "out"
    dest.mkdir()
    extract_csvs(outer, dest)
    assert sorted(p.name for p in dest.iterdir()) == ["a.csv", "b.csv"]
