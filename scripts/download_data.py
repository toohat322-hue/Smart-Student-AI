"""Download the UCI Student Performance dataset (#320) into data/.

The outer ZIP from UCI contains an inner ``student.zip`` (plus ``student.txt``
and ``student-merge.R``); the two CSVs live inside that inner archive.

Steps:
  1. Skip everything if data/student-mat.csv and data/student-por.csv exist.
  2. Download the outer ZIP into data/raw/.
  3. Open the nested student.zip and copy the two CSVs into data/.

Usage:
    python scripts/download_data.py            # skip if CSVs already exist
    python scripts/download_data.py --force    # re-download and overwrite
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR, SUBJECTS  # noqa: E402

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
RAW_DIR = DATA_DIR / "raw"
OUTER_ZIP = RAW_DIR / "student+performance.zip"
INNER_ZIP_NAME = "student.zip"
CSV_NAMES = sorted(SUBJECTS.values())  # student-mat.csv, student-por.csv
TIMEOUT_S = 60


def csvs_present() -> bool:
    """True if both subject CSVs are already in data/."""
    return all((DATA_DIR / name).is_file() for name in CSV_NAMES)


def download(url: str, dest: Path) -> None:
    """Download a URL to dest, with clear errors if UCI cannot be reached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "smart-student-ai-downloader"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"ERROR: UCI returned HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SystemExit(
            f"ERROR: could not reach UCI ({exc}).\n"
            f"Download the ZIP manually from {url}\n"
            f"and place {', '.join(CSV_NAMES)} in {DATA_DIR}"
        ) from exc
    if not zipfile.is_zipfile(io.BytesIO(payload)):
        raise SystemExit(f"ERROR: downloaded file from {url} is not a valid ZIP archive")
    dest.write_bytes(payload)
    print(f"Saved {dest.relative_to(ROOT_DIR)} ({len(payload):,} bytes)")


def _find_member(zf: zipfile.ZipFile, name: str) -> str:
    """Return the archive member whose basename is ``name`` (ignores sub-folders)."""
    for member in zf.namelist():
        if Path(member).name == name:
            return member
    raise SystemExit(f"ERROR: '{name}' not found in archive (members: {zf.namelist()})")


def extract_csvs(outer_zip: Path) -> None:
    """Read the nested ZIP in memory and write only the two expected CSVs.

    Members are written by their basename into DATA_DIR, so archive paths can
    never escape the data folder.
    """
    with zipfile.ZipFile(outer_zip) as outer:
        inner_bytes = outer.read(_find_member(outer, INNER_ZIP_NAME))
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        for name in CSV_NAMES:
            data = inner.read(_find_member(inner, name))
            (DATA_DIR / name).write_bytes(data)
            print(f"Extracted data/{name} ({len(data):,} bytes)")


def main(argv: list[str] | None = None) -> int:
    """Download and extract the two CSVs unless they are already present."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if CSVs exist")
    args = parser.parse_args(argv)

    if csvs_present() and not args.force:
        print(f"Both CSVs already present in {DATA_DIR} - skipping download.")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTER_ZIP.is_file() or args.force:
        download(UCI_ZIP_URL, OUTER_ZIP)
    else:
        print(f"Using cached {OUTER_ZIP.relative_to(ROOT_DIR)}")

    try:
        extract_csvs(OUTER_ZIP)
    except zipfile.BadZipFile as exc:
        raise SystemExit(f"ERROR: corrupt archive {OUTER_ZIP} ({exc}); rerun with --force") from exc

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
