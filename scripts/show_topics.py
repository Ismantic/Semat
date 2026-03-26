#!/usr/bin/env python3
"""
Display top words per topic from semat.phi

Usage: python show_topics.py <phi_file> [topn=30]
"""

import sys

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <phi_file> [topn=30]", file=sys.stderr)
        sys.exit(1)

    phi_file = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    with open(phi_file) as f:
        topic = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Topic "):
                topic = line
                continue
            pairs = line.split()
            words = []
            for p in pairs:
                w, _ = p.rsplit(":", 1)
                words.append(w)
                if len(words) >= topn:
                    break
            print(f"{topic} {' '.join(words)}")

if __name__ == "__main__":
    main()
