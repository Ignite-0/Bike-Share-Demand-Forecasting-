"""Download the last N months of Citi Bike trip zips and extract CSVs to data/raw/trips/YYYYMM/.

Idempotent: a month with a .done marker is skipped. Zips are deleted after extraction to save disk.
Usage: python -m src.ingest.download_trips [--months 12]
"""
import argparse
import re
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

BUCKET = "https://s3.amazonaws.com/tripdata/"
OUT = Path("data/raw/trips")
# NYC monthly files only (excludes JC-*, yearly archives). Some months use ".csv.zip".
MONTHLY = re.compile(r"^(\d{6})-citibike-tripdata(\.csv)?\.zip$")


def list_monthly_keys() -> dict[str, str]:
    """Return {YYYYMM: key} for every monthly NYC file in the bucket."""
    keys, token = {}, None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        url = BUCKET + "?list-type=2" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        for k in root.iterfind("s3:Contents/s3:Key", ns):
            if m := MONTHLY.match(k.text):
                keys[m.group(1)] = k.text
        token = root.findtext("s3:NextContinuationToken", namespaces=ns)
        if not token:
            return keys


def extract_csvs(zip_path: Path, dest: Path) -> None:
    """Extract every CSV (also from nested zips) flat into dest, skipping __MACOSX junk."""
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if info.is_dir() or info.filename.startswith("__MACOSX") or name.startswith("._"):
                continue
            if name.endswith(".csv"):
                with z.open(info) as src, open(dest / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            elif name.endswith(".zip"):
                nested = dest / name
                with z.open(info) as src, open(nested, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extract_csvs(nested, dest)
                nested.unlink()


def main(months: int) -> None:
    keys = list_monthly_keys()
    for ym in sorted(keys)[-months:]:
        dest = OUT / ym
        if (dest / ".done").exists():
            print(f"{ym}: already done")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / keys[ym]
        print(f"{ym}: downloading {keys[ym]}")
        tmp = zip_path.with_suffix(".part")
        urllib.request.urlretrieve(BUCKET + keys[ym], tmp)  # .part so a crash never looks complete
        tmp.replace(zip_path)
        extract_csvs(zip_path, dest)
        zip_path.unlink()
        (dest / ".done").touch()
        print(f"{ym}: {len(list(dest.glob('*.csv')))} csv files")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=12)
    main(p.parse_args().months)
