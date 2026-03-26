#!/usr/bin/env python3
"""
TF-IDF reweighting for LDA corpus.

For each word in each document:
  score = log(1 + TF) * IDF
  where IDF = log(N / DF)
  repeat the word round(score) times; drop if score < threshold.

Usage:
  python tfidf_reweight.py <input> <output> [--threshold 0.5]
"""

import sys
import math
from collections import Counter

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input> <output> [--threshold 0.5]", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    threshold = 0.5

    for i in range(3, len(sys.argv)):
        if sys.argv[i] == "--threshold" and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])

    # Pass 1: compute DF
    print("Pass 1: computing DF...", file=sys.stderr)
    df = Counter()
    N = 0
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = set(line.split())
            for w in words:
                df[w] += 1
            N += 1

    print(f"  {N} documents, {len(df)} unique words", file=sys.stderr)

    # Pass 2: reweight and write
    print(f"Pass 2: reweighting (threshold={threshold})...", file=sys.stderr)
    total_before = 0
    total_after = 0
    dropped_words = 0

    with open(input_file) as fin, open(output_file, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            tf = Counter(words)
            total_before += len(words)

            out = []
            for w in words:
                idf = math.log10(N / df[w])
                score = math.log10(tf[w]) * idf if tf[w] > 1 else 0
                count = round(score)
                if score < threshold:
                    dropped_words += 1
                    continue
                out.append(w)
                for _ in range(count - 1):
                    out.append(w)

            total_after += len(out)
            if out:
                fout.write(" ".join(out) + "\n")

    print(f"  tokens: {total_before} -> {total_after}", file=sys.stderr)
    print(f"  dropped: {dropped_words} token occurrences below threshold", file=sys.stderr)

if __name__ == "__main__":
    main()
