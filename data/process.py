#!/usr/bin/env python3
"""Convert THUCNews into a corpus with one complete article per line."""

import argparse
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "downloads" / "THUCNews" / "default" / "train"
OUTPUT = HERE / "derived" / "THUCNews.documents.txt"
SPACE = re.compile(r"\s+")


def clean(value):
    return SPACE.sub(" ", value or "").strip()


def document(title, text):
    return " ".join(part for part in (clean(title), clean(text)) if part)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--max-documents",
        type=int,
        help="stop after this many documents (for development smoke runs)",
    )
    return parser.parse_args()


def main():
    import pyarrow.parquet as pq

    args = parse_args()
    if args.max_documents is not None and args.max_documents < 1:
        raise SystemExit("--max-documents must be positive")
    files = sorted(args.input_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(
            f"No Parquet files under {args.input_dir}; run data/download.py"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    rows = documents = 0
    finished = False
    with temporary.open("w", encoding="utf-8") as output:
        for path in files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=2048, columns=["title", "text"]
            ):
                titles = batch.column("title").to_pylist()
                texts = batch.column("text").to_pylist()
                for title, text in zip(titles, texts):
                    rows += 1
                    line = document(title, text)
                    if line:
                        output.write(line + "\n")
                        documents += 1
                        if (
                            args.max_documents is not None
                            and documents >= args.max_documents
                        ):
                            finished = True
                            break
                if finished:
                    break
            print(f"{path.name}: {rows:,} rows, {documents:,} documents")
            if finished:
                break
    os.replace(temporary, args.output)
    print(
        f"Wrote {documents:,} documents from {rows:,} rows -> {args.output}"
    )


if __name__ == "__main__":
    main()
