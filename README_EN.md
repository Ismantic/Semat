# Semat

English | [中文](README.md)

Semat (Sparse Efficient Model Allocation Topic) is a concise and practical
C++17 LDA implementation. Its core algorithm uses SparseLDA's S/R/Q bucket
sampler and N-Queen document-word partitioning for parallel Gibbs sampling.

## Features

- Sparse document-topic and word-topic counts
- N-Queen multithreaded scheduling
- Initialization from Wavec word-vector K-means clusters
- Document-level TF-IDF corpus reduction
- Training perplexity and active-topic monitoring
- `.vocab`, `.phi`, and `.theta` model outputs

## Pipeline

Semat and [Wavec](https://github.com/Ismantic/Wavec) use the same data source
and Wapic segmenter, but their training units differ. Wavec trains CBOW on one
sentence per line; Semat must preserve one complete article per line.

```text
News Parquet
  → merge title and body (one article per line)
  → document-level Wapic segmentation
  → DF counting and TF-IDF reweighting
  → initialize topics from a Wavec K-means mapping
  → SparseLDA / N-Queen training
```

`data/download.py` downloads `SirlyDreamer/THUCNews` from Hugging Face.
`data/process.py` creates `data/derived/THUCNews.documents.txt`, preserving
document boundaries throughout preprocessing.

## Requirements and Build

Semat requires CMake 3.14+, a C++17 compiler, and Python 3.9+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The resulting executable is `build/semat`.

## Wavec Initialization

Training requires a Wavec K-means mapping whose cluster count matches the
requested topic count. By default, Semat reads the pure-Chinese 100-cluster
mapping from the adjacent Wavec checkout:

```text
../Wavec/data/wavec.20260405.Kmeans.map
```

Each mapping line may use either `word cluster_id` or
`word<TAB>cluster_id`. Semat verifies that the number of covered clusters
equals `TOPICS`. TF-IDF conversion also restricts the training vocabulary to
mapped words, preventing numbers, URLs, and codes from forming noisy topics.

Specify another Wavec checkout or mapping with:

```bash
make -C scripts fit WAVEC_ROOT=/path/to/Wavec
make -C scripts fit INIT=/path/to/clusters.map TOPICS=100
```

The pipeline stops when the mapping is absent. Run word-vector training,
filtering, and K-means in Wavec first.

## Running

After preparing the Wavec mapping, run the complete Semat download,
document-conversion, segmentation, filtering, training, and topic-display
pipeline with:

```bash
make -C scripts all NPROC=8 THREADS=8
```

Run individual stages when needed:

```bash
make -C scripts data
make -C scripts cut NPROC=8
make -C scripts count
make -C scripts conv MIN_DF=10 MIN_LEN=2 MIN_SCORE=2.0 MIN_UNIQ=10
make -C scripts fit TOPICS=100 ITERS=150 THREADS=8
make -C scripts print TOPN=30
```

Default outputs are:

- `scripts/output/semat.vocab`: training vocabulary
- `scripts/output/semat.phi`: word probabilities for each topic
- `scripts/output/semat.theta`: topic probabilities for each document

Override paths and local settings with `RAW_CORPUS`, `SEG_FILE`,
`TRAIN_CORPUS`, `OUTPUT`, or the ignored `local.mk` file.

## Example Topics

The following top words come from a THUCNews run with 100 topics and 150
sampling iterations:

| Topic | Top words |
|---|---|
| Film and TV | 导演, 演员, 春晚, 明星, 剧组, 电影, 赵本山, 华谊, 微博, 拍摄 |
| Basketball | 火箭, 球队, 湖人, 赛季, 科比, 球员, 篮板, 詹姆斯, 热火, 火箭队 |
| Education | 考生, 招生, 高考, 录取, 考试, 学校, 学生, 高校, 志愿, 报考 |
| Healthcare | 医院, 医生, 手术, 医疗, 治疗, 检查, 药品, 死亡, 抢救, 卫生 |
| Real estate | 项目, 房地产, 城市, 平方米, 土地, 面积, 房价, 地产, 开发商, 住宅 |
| Financial markets | 指数, 板块, 上涨, 下跌, 反弹, 市场, 股市, 涨幅, 震荡, 资金 |

Results depend on the corpus, cluster initialization, and random sampling.
Topic IDs and word order are not guaranteed to remain identical across runs.

## CLI

```bash
./build/semat <corpus> <topics> <iters> <alpha> <beta> <threads> \
  --init <clusters.map> --output <prefix>
```

For example:

```bash
./build/semat scripts/News.dat.txt 100 150 0.1 0.01 8 \
  --init ../Wavec/data/wavec.20260405.Kmeans.map \
  --output scripts/output/semat
```

## Tests

```bash
make -C scripts test
# or
ctest --test-dir build --output-on-failure
```

The smoke test uses temporary documents and covers DF counting, TF-IDF
conversion, cluster initialization, multithreaded training, model output, and
topic display without downloading THUCNews.

## Notes

The trainer loads the reduced corpus into memory. N-Queen scheduling prevents
threads from modifying the same document or word concurrently, while shared
topic totals use atomic operations. Perplexity monitors the training trend;
model quality should also be evaluated through topic readability and
downstream tasks.

See [LDA and SparseLDA](https://ismantic.github.io/text/semat.html) for an
algorithm walkthrough.

## License

MIT. Refer to the upstream projects for the licensing terms of THUCNews,
Wapic, and Wavec.
