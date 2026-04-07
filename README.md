# Semat

Sparse Efficient Model Allocation Topic — 基于 SparseLDA + N-Queen 多线程调度的 LDA 主题模型实现。

## 特性

- **SparseLDA 采样**：将采样概率分解为 s/r/q 三个桶，降低采样复杂度
- **N-Queen 并行**：文档-词汇分块调度（modulo 分配），无锁多线程 Gibbs 采样
- **稀疏计数**：使用 `unordered_map` 存储计数矩阵，适合大规模稀疏数据
- **KMeans 初始化**：支持从词向量聚类结果初始化 topic 分配，加速收敛
- **Perplexity 监控**：训练过程中输出困惑度，评估模型收敛

## 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## 数据准备

将以下文件放入 `prepare/` 目录：

- `News.cut.txt` — 分词后语料，每行一篇文章，词语空格分隔，切词工具 [Iscut](https://github.com/Ismantic/Iscut)
- `wavec.20260405.Kmeans.map` — 词向量聚类映射（`word\tclusterID`，来自 [Wavec](https://github.com/Ismantic/Wavec)）

## 使用

通过 `scripts/Makefile` 驱动完整流程：

```bash
make -C scripts count    # 统计词频（DF）
make -C scripts conv     # TF-IDF 加权转换（若未统计则自动 count）
make -C scripts fit      # 训练（若未转换则自动 conv）
make -C scripts print    # 查看主题 top 词
```

```bash
make -C scripts fit TOPICS=128 ITERS=200 THREADS=16
make -C scripts conv MIN_DF=20 MIN_SCORE=3.0
```

### 输出文件

训练完成后生成三个文件：

- `<output>.vocab` — 词汇表
- `<output>.phi` — 主题-词语分布（每个主题的 top 词语及概率）
- `<output>.theta` — 文档-主题分布

## 工具

### process.py — 语料处理

```bash
python3 scripts/process.py count <seg_file> <output>       # 统计 DF
python3 scripts/process.py conv <seg_file> <vocab> <output> # TF-IDF 加权
```

`conv` 对每个文档中的词计算 `score = log(TF) * log(N/DF)`，按分数重复词语，过滤低质量词和文档。

### print_topics.py — 主题查看

```bash
python3 scripts/print_topics.py <phi_file> [topn=30]
```

## License

MIT
