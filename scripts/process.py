#!/usr/bin/env python3
"""Corpus processing tools for Semat.

Commands:
  vocab   - Count document frequency (DF) for each word
  tfidf   - TF-IDF reweight corpus with filtering
"""

import sys
import math
from collections import Counter


def vocab(seg_file, output_file):
    """Count DF and output word\\tDF sorted by DF descending."""
    df = Counter()
    n_docs = 0

    with open(seg_file) as f:
        for line in f:
            words = set(line.split())
            if not words:
                continue
            for w in words:
                df[w] += 1
            n_docs += 1
            if n_docs % 1000000 == 0:
                print(f"  {n_docs} docs...", file=sys.stderr)

    with open(output_file, "w") as f:
        for w, c in df.most_common():
            f.write(f"{w}\t{c}\n")

    print(f"Done: {n_docs} docs, {len(df)} words -> {output_file}", file=sys.stderr)


def tfidf(seg_file, vocab_file, output_file, min_df=10, min_len=2, min_score=2.0, min_uniq=10):
    """TF-IDF reweight corpus.

    For each word in each document:
      score = log(TF) * log(N / DF)

    Filtering:
      - word length < min_len: drop
      - DF < min_df: drop
      - score < min_score: drop
      - unique words in doc < min_uniq: drop doc
    """
    # Load DF from vocab file
    df = {}
    with open(vocab_file) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            df[parts[0]] = int(parts[1])

    n_docs = sum(1 for _ in open(seg_file))
    print(f"N={n_docs}, vocab={len(df)} words", file=sys.stderr)

    # Filter vocab
    keep = {w for w, c in df.items() if c >= min_df and len(w) >= min_len}
    print(f"After filter (min_df={min_df}, min_len={min_len}): {len(keep)} words", file=sys.stderr)

    total_before = 0
    total_after = 0
    n_out = 0

    with open(seg_file) as fin, open(output_file, "w") as fout:
        for line in fin:
            words = line.split()
            if not words:
                continue
            tf = Counter(w for w in words if w in keep)
            total_before += len(words)

            out = []
            for w, count in tf.items():
                if count <= 1:
                    continue
                score = math.log(count) * math.log(n_docs / df[w])
                if score < min_score:
                    continue
                rep = round(score)
                if rep < 1:
                    rep = 1
                out.extend([w] * rep)

            uniq_words = set(out)
            if len(uniq_words) >= min_uniq:
                fout.write(" ".join(out) + "\n")
                n_out += 1
            total_after += len(out)

    print(f"Docs: {n_docs} -> {n_out}", file=sys.stderr)
    print(f"Tokens: {total_before} -> {total_after}", file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        print("Usage:", file=sys.stderr)
        print(f"  {sys.argv[0]} vocab <seg_file> <output>", file=sys.stderr)
        print(f"  {sys.argv[0]} tfidf <seg_file> <vocab_file> <output> [--min-df 10] [--min-len 2] [--min-score 1.0] [--min-uniq 10]", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "vocab":
        if len(sys.argv) < 4:
            print(f"Usage: {sys.argv[0]} vocab <seg_file> <output>", file=sys.stderr)
            sys.exit(1)
        vocab(sys.argv[2], sys.argv[3])

    elif cmd == "tfidf":
        if len(sys.argv) < 5:
            print(f"Usage: {sys.argv[0]} tfidf <seg_file> <vocab_file> <output> [--min-df 10] [--min-len 2] [--min-score 1.0] [--min-uniq 10]", file=sys.stderr)
            sys.exit(1)
        seg_file = sys.argv[2]
        vocab_file = sys.argv[3]
        output_file = sys.argv[4]
        min_df = 10
        min_len = 2
        min_score = 1.0
        min_uniq = 10
        i = 5
        while i < len(sys.argv):
            if sys.argv[i] == "--min-df" and i + 1 < len(sys.argv):
                min_df = int(sys.argv[i + 1]); i += 2
            elif sys.argv[i] == "--min-len" and i + 1 < len(sys.argv):
                min_len = int(sys.argv[i + 1]); i += 2
            elif sys.argv[i] == "--min-score" and i + 1 < len(sys.argv):
                min_score = float(sys.argv[i + 1]); i += 2
            elif sys.argv[i] == "--min-uniq" and i + 1 < len(sys.argv):
                min_uniq = int(sys.argv[i + 1]); i += 2
            else:
                i += 1
        tfidf(seg_file, vocab_file, output_file, min_df, min_len, min_score, min_uniq)

    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
