# THUCNews data

This directory uses the same THUCNews source as Wavec, but preserves Semat's
document-level training semantics. Each Parquet article becomes one output line;
the title and body are joined after embedded whitespace is normalized.

```bash
make -C data status
make -C data download
make -C data process
```

Downloaded Parquet files are stored under `downloads/`; the document corpus is
written to `derived/THUCNews.documents.txt`. Both directories are ignored by Git.
