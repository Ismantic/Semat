#!/usr/bin/env python3
"""Download the THUCNews Parquet export from Hugging Face."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "SirlyDreamer/THUCNews"
REVISION = "refs/convert/parquet"
HERE = Path(__file__).resolve().parent
DEST = HERE / "downloads" / "THUCNews"


def remote_files():
    api = HfApi()
    return sorted(
        name
        for name in api.list_repo_files(
            REPO_ID, repo_type="dataset", revision=REVISION
        )
        if name.startswith("default/train/") and name.endswith(".parquet")
    )


def status(files):
    present = sum((DEST / name).is_file() for name in files)
    size = sum(
        (DEST / name).stat().st_size
        for name in files
        if (DEST / name).is_file()
    )
    print(f"THUCNews: {present}/{len(files)} files, {size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    files = remote_files()
    if not files:
        raise SystemExit(f"No Parquet files found in {REPO_ID}@{REVISION}")
    if args.status:
        status(files)
        return

    for index, name in enumerate(files, 1):
        print(f"[{index}/{len(files)}] {name}", flush=True)
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            filename=name,
            local_dir=DEST,
        )
    status(files)


if __name__ == "__main__":
    main()
