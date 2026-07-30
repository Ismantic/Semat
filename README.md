# Semat

[English](README_EN.md) | 中文

Semat（Sparse Efficient Model Allocation Topic）是一个兼具简洁与实用的
C++17 LDA 实现。核心算法采用 SparseLDA 的 S/R/Q 三桶采样，并通过 N-Queen
文档—词汇分块并行执行 Gibbs 采样。

## 特性

- 稀疏文档—主题和词—主题计数
- N-Queen 多线程调度
- Wavec 词向量 K-means 初始化
- 文档级 TF-IDF 语料压缩
- 训练困惑度和活跃主题监控
- `.vocab`、`.phi`、`.theta` 模型输出

## 流程

Semat 与 [Wavec](https://github.com/Ismantic/Wavec) 使用相同的数据源和
Wapic 分词器，但训练单位不同：Wavec 的 CBOW 输入是一行一句，Semat 的输入必须
保持一行一篇文章。

```text
News Parquet
  → 合并标题和正文（一行一篇文章）
  → Wapic 文档级分词
  → DF 统计和 TF-IDF 重加权
  → 使用 Wavec K-means 映射初始化 topic
  → SparseLDA / N-Queen 训练
```

`data/download.py` 从 Hugging Face 的 `SirlyDreamer/THUCNews` 下载数据，
`data/process.py` 生成 `data/derived/THUCNews.documents.txt`。预处理过程中始终
保留文档边界。

## 环境与构建

需要 CMake 3.14+、C++17 编译器和 Python 3.9+。

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

构建产物为 `build/semat`。

## Wavec 初始化

训练前必须准备与主题数一致的 Wavec K-means 映射。默认读取相邻仓库中的纯中文
100 类映射：

```text
../Wavec/data/wavec.20260405.Kmeans.map
```

映射格式可以是 `word cluster_id` 或 `word<TAB>cluster_id`。Semat 会检查映射覆盖
的聚类数是否等于 `TOPICS`，并在 TF-IDF 阶段仅保留映射词表内的词，避免数字、
URL 和代码形成无效主题。

使用其他 Wavec 仓库或映射时传入：

```bash
make -C scripts fit WAVEC_ROOT=/path/to/Wavec
make -C scripts fit INIT=/path/to/clusters.map TOPICS=100
```

映射不存在时流程会停止；请先在 Wavec 中完成词向量训练、过滤和 K-means。

## 运行

确认 Wavec 映射存在后，一条命令完成 Semat 的数据下载、文档转换、分词、过滤、
训练和主题打印：

```bash
make -C scripts all NPROC=8 THREADS=8
```

也可以逐阶段执行：

```bash
make -C scripts data
make -C scripts cut NPROC=8
make -C scripts count
make -C scripts conv MIN_DF=10 MIN_LEN=2 MIN_SCORE=2.0 MIN_UNIQ=10
make -C scripts fit TOPICS=100 ITERS=150 THREADS=8
make -C scripts print TOPN=30
```

默认输出位于：

- `scripts/output/semat.vocab`：训练词表
- `scripts/output/semat.phi`：每个主题的词概率
- `scripts/output/semat.theta`：每篇文档的主题概率

路径和本机参数可通过 `RAW_CORPUS`、`SEG_FILE`、`TRAIN_CORPUS`、`OUTPUT` 或
忽略提交的 `local.mk` 覆盖。

## Topic

以下是使用 THUCNews、100 个主题和 150 轮采样得到的部分 Top 词：

| ID | Top words |
|---|---|
| 13 | 导演、演员、春晚、明星、剧组、电影、赵本山、华谊、微博、拍摄 |
| 20 | 火箭、球队、湖人、赛季、科比、球员、篮板、詹姆斯、热火、火箭队 |
| 73 | 考生、招生、高考、录取、考试、学校、学生、高校、志愿、报考 |
| 97 | 医院、医生、手术、医疗、治疗、检查、药品、死亡、抢救、卫生 |
| 83 | 项目、房地产、城市、平方米、土地、面积、房价、地产、开发商、住宅 |
| 77 | 指数、板块、上涨、下跌、反弹、市场、股市、涨幅、震荡、资金 |

实际结果取决于语料、聚类初始化和随机采样；topic ID 与词语顺序不保证在不同训练间
完全一致。

## 调用

```bash
./build/semat <corpus> <topics> <iters> <alpha> <beta> <threads> \
  --init <clusters.map> --output <prefix>
```

例如：

```bash
./build/semat scripts/News.dat.txt 100 150 0.1 0.01 8 \
  --init ../Wavec/data/wavec.20260405.Kmeans.map \
  --output scripts/output/semat
```

## 测试

```bash
make -C scripts test
# 或
ctest --test-dir build --output-on-failure
```

冒烟测试使用临时文档，覆盖 DF 统计、TF-IDF 转换、聚类初始化、多线程训练、模型
输出和主题打印，不需要下载 THUCNews。

## 说明

训练器会把压缩后的语料载入内存。N-Queen 调度避免线程同时修改同一文档或同一词，
共享主题总计数使用原子操作。困惑度用于观察训练趋势；正式模型质量还应结合主题词
可读性和下游任务评估。

算法说明见[番外篇：LDA 与 SparseLDA](https://ismantic.github.io/text/semat.html)。

## License

MIT。THUCNews、Wapic 和 Wavec 的许可条件请参考各自上游项目。
