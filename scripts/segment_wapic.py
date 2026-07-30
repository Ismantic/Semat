#!/usr/bin/env python3
"""Segment a line-oriented corpus with Wapic."""

import argparse
import os
import sys

import wapic


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="UTF-8 corpus, one sentence per line")
    parser.add_argument("output", help="space-separated segmented corpus")
    parser.add_argument("--model", help="Wapic .wac model; omit to use the bundled model")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def write_batch(segmenter, lines, output):
    for words in segmenter.segment_batch(lines):
        output.write(" ".join(words))
        output.write("\n")


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    segmenter = wapic.Segmenter(args.model) if args.model else wapic.Segmenter()
    processed = 0
    next_report = 1_000_000
    batch = []

    temporary = args.output + ".tmp"
    with open(args.input, encoding="utf-8") as source, open(
        temporary, "w", encoding="utf-8"
    ) as output:
        for line in source:
            batch.append(line.rstrip("\r\n"))
            if len(batch) == args.batch_size:
                write_batch(segmenter, batch, output)
                processed += len(batch)
                if processed >= next_report:
                    print(f"Segmented {processed:,} lines", file=sys.stderr)
                    next_report += 1_000_000
                batch.clear()
        if batch:
            write_batch(segmenter, batch, output)
            processed += len(batch)

    os.replace(temporary, args.output)
    print(f"Segmented {processed:,} lines -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
