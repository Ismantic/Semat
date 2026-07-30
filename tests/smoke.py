#!/usr/bin/env python3
"""Small end-to-end test for corpus processing and the Semat CLI."""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path


SEMAT, PROCESS, PRINT_TOPICS, DATA_PROCESS = sys.argv[1:5]

spec = importlib.util.spec_from_file_location("data_process", DATA_PROCESS)
data_process = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_process)
assert data_process.document(" 新闻标题\n", "第一段。\r\n第二段。") == (
    "新闻标题 第一段。 第二段。"
)


def run(*args, expected=0):
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    if result.returncode != expected:
        raise AssertionError(
            f"{' '.join(args)} returned {result.returncode}, expected {expected}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    segmented = root / "tiny.cut.txt"
    vocab = root / "count.txt"
    corpus = root / "tiny.dat.txt"
    mapping = root / "clusters.map"
    output = root / "semat"
    segmented.write_text(
        ("北京 北京 上海 上海 城市 城市 新闻 新闻 中国 中国\n") * 4,
        encoding="utf-8",
    )
    mapping.write_text("北京\t0\n上海\t1\n城市\t2\n新闻\t3\n中国\t0\n", encoding="utf-8")

    run(sys.executable, PROCESS, "count", str(segmented), str(vocab))
    run(
        sys.executable,
        PROCESS,
        "conv",
        str(segmented),
        str(vocab),
        str(corpus),
        "--min-df", "1",
        "--min-len", "1",
        "--min-score", "0",
        "--min-uniq", "1",
        "--allowed-vocab", str(mapping),
    )
    run(
        SEMAT,
        str(corpus),
        "4",
        "1",
        "0.1",
        "0.01",
        "2",
        "--init",
        str(mapping),
        "--output",
        str(output),
    )
    for suffix in (".vocab", ".phi", ".theta"):
        assert Path(str(output) + suffix).is_file()
    topics = run(sys.executable, PRINT_TOPICS, str(output) + ".phi", "3")
    assert "Topic " in topics.stdout
    run(SEMAT, str(corpus), "4", "1", "0.1", "0.01", "2", expected=1)
    run(
        SEMAT,
        str(corpus),
        "5",
        "1",
        "0.1",
        "0.01",
        "2",
        "--init",
        str(mapping),
        expected=1,
    )
