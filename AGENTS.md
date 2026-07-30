# Repository Guidelines

## Project Structure & Module Organization

`src/semat.cc` contains the C++17 trainer. `data/` contains THUCNews acquisition code. `scripts/` contains Wapic segmentation, corpus weighting, topic inspection, and the pipeline Makefile. `tests/smoke.py` covers the CLI workflow; `data/semat.phi` is sample output. Generated inputs belong in ignored data and `prepare/` directories; models belong in `scripts/output/`. Preserve one article per line: a line is Semat's document unit.

## Build, Test, and Development Commands

- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` configures the C++ build.
- `cmake --build build` compiles `build/semat`.
- `make -C scripts data` downloads and converts THUCNews.
- `make -C scripts cut` segments the corpus with Wapic.
- `make -C scripts count` computes document frequencies from `prepare/THUCNews.cut.txt`.
- `make -C scripts conv` creates the TF-IDF-weighted training corpus.
- `make -C scripts fit TOPICS=100 ITERS=150 THREADS=8` trains the default model.
- `make -C scripts print TOPN=30` displays the highest-ranked words per topic.
- `make -C scripts all` runs the complete pipeline.

Training requires Wavec K-means initialization. The default is the sibling
Wavec repository's checked-in pure-Chinese mapping; override `WAVEC_ROOT` or
`INIT` for another checkout. Generate a missing mapping in Wavec first.

## Coding Style & Naming Conventions

Use four-space indentation in C++ and Python; use tabs only in Makefile recipes. Keep C++ compatible with C++17. Follow existing `PascalCase` C++ methods and `snake_case` Python names. Prefer standard-library facilities and keep errors on `stderr`. No formatter or linter is configured, so match nearby code.

## Testing Guidelines

Run `make -C scripts test` or `ctest --test-dir build --output-on-failure`. The smoke test covers conversion, initialized training, model outputs, and topic printing without downloading THUCNews. Also exercise pipeline changes with a small real corpus.

## Commit & Pull Request Guidelines

History uses brief, imperative subjects such as `Update README.md`; retain concise subjects, but identify the affected area when useful (for example, `Fix corpus filtering threshold`). Keep each commit scoped to one logical change. Pull requests should explain the motivation, list validation commands, note performance implications, and link related issues. Include terminal output for training changes and topic samples when model behavior changes.

## Data and Configuration

Do not commit downloaded corpora, segmented data, models, or machine-specific paths. Put local overrides in ignored `local.mk`. Expose tunable values through Make variables or CLI flags. Preserve temporary-file-then-replace behavior for large generated data so interrupted runs do not leave apparently complete outputs.
