"""
Automatically download the freely available HAR datasets.

Datasets downloaded:
  - HAPT   (UCI, ~50 MB)
  - PAMAP2 (UCI, ~650 MB)
  - WISDM  (~3 MB)

MobiAct requires registration at https://bmi.hmu.gr/the-mobiact-dataset-v2-0/
→ download manually and place under data/raw/mobiact/

Usage:
    python scripts/download_datasets.py --data_root data/raw
    python scripts/download_datasets.py --data_root data/raw --datasets hapt wisdm
"""

import argparse
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "hapt": (
        "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip",
        "hapt.zip",
    ),
    "pamap2": (
        "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
        "pamap2.zip",
    ),
    "wisdm": (
        "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz",
        "wisdm.tar.gz",
    ),
}


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct / 2)
        print(f"\r  [{bar:<50s}] {pct:5.1f}%", end="", flush=True)


def download_file(url: str, dest: Path, filename: str):
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / filename
    if out_path.exists():
        print(f"  Already downloaded: {out_path}")
        return out_path
    print(f"  Downloading {filename} ...")
    try:
        urllib.request.urlretrieve(url, out_path, reporthook=_progress)
        print()
    except Exception as e:
        print(f"\n  Failed: {e}")
        if out_path.exists():
            out_path.unlink()
        return None
    return out_path


def extract_zip(zip_path: Path, dest: Path):
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def extract_targz(tgz_path: Path, dest: Path):
    import tarfile
    print(f"  Extracting {tgz_path.name} ...")
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(dest)


def download_hapt(raw_root: Path):
    folder = raw_root / "hapt"
    # Check if already extracted
    if (folder / "RawData").exists() or list(folder.glob("**/labels.txt")):
        print("  HAPT already extracted.")
        return

    url, fname = URLS["hapt"]
    zip_path = download_file(url, raw_root / "_tmp", fname)
    if zip_path is None:
        return

    # The zip contains a nested zip — extract outer then inner
    outer_dir = raw_root / "_tmp" / "hapt_outer"
    extract_zip(zip_path, outer_dir)

    # Find the inner zip
    inner_zips = list(outer_dir.rglob("*.zip"))
    if inner_zips:
        extract_zip(inner_zips[0], folder)
    else:
        # Sometimes it's flat
        import shutil
        shutil.copytree(str(outer_dir), str(folder), dirs_exist_ok=True)

    print(f"  HAPT ready at {folder}")


def download_pamap2(raw_root: Path):
    folder = raw_root / "pamap2"
    if list(folder.rglob("subject10*.dat")):
        print("  PAMAP2 already extracted.")
        return

    url, fname = URLS["pamap2"]
    zip_path = download_file(url, raw_root / "_tmp", fname)
    if zip_path is None:
        return

    extract_zip(zip_path, folder)
    print(f"  PAMAP2 ready at {folder}")


def download_wisdm(raw_root: Path):
    folder = raw_root / "wisdm"
    if list(folder.glob("*.txt")):
        print("  WISDM already extracted.")
        return

    url, fname = URLS["wisdm"]
    tgz_path = download_file(url, raw_root / "_tmp", fname)
    if tgz_path is None:
        # Fallback: try direct txt file
        fallback = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
        tgz_path = download_file(fallback, raw_root / "_tmp", fname)

    if tgz_path is None:
        return

    folder.mkdir(parents=True, exist_ok=True)
    extract_targz(tgz_path, folder)
    print(f"  WISDM ready at {folder}")


DOWNLOADERS = {
    "hapt":   download_hapt,
    "pamap2": download_pamap2,
    "wisdm":  download_wisdm,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--datasets", nargs="*",
                        default=["hapt", "pamap2", "wisdm"],
                        choices=list(DOWNLOADERS))
    args = parser.parse_args()

    raw_root = Path(args.data_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading datasets to: {raw_root.resolve()}\n")

    for name in args.datasets:
        print(f"=== {name.upper()} ===")
        DOWNLOADERS[name](raw_root)
        print()

    # Cleanup temp
    import shutil
    tmp = raw_root / "_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    print("Done. Next: python scripts/preprocess.py --data_root", args.data_root)
    print()
    print("Note: MobiAct requires manual download from:")
    print("  https://bmi.hmu.gr/the-mobiact-dataset-v2-0/")
    print("  → extract to data/raw/mobiact/")


if __name__ == "__main__":
    main()
