# Semat 训练指南

本文档记录从原始语料到高质量 LDA 主题模型的完整训练流程。

## 整体流程

```
原始文本 → 分词 → 单字过滤 → TF-IDF 加权 → 词向量训练 → KMeans 聚类 → LDA 训练
```

## 1. 语料准备

### 1.1 分词

输入文件要求每行一个文档，词语空格分隔。中文语料需先分词：

```bash
# 使用 IsmaCut 分词
ismacut < raw.txt > segmented.txt
```

### 1.2 单字过滤

去除单字词（"的"、"了"、"在"等），减少噪声：

```bash
awk '{out=""; for(i=1;i<=NF;i++) if(length($i)>3) out=out" "$i; if(out!="") print substr(out,2)}' \
    segmented.txt > filtered.txt
```

> 中文 UTF-8 编码下，单个汉字占 3 字节，`length > 3` 即过滤单字。

## 2. TF-IDF 语料加权

用 TF-IDF 对语料重新加权，使 LDA 更关注有区分度的词：

```bash
python3 scripts/tfidf_reweight.py <input> <output> [--threshold 1]
```

**原理**：对文档中每个词计算 `score = log10(TF) * log10(N/DF)`：
- TF=1 的词（只出现一次），score=0，被截断
- 高 TF + 高 IDF（文档内高频且有区分度）的词被保留或重复
- 高 DF 的通用词（"表示"、"可以"）因 IDF 低而被压制

**阈值选择**：
- `threshold=0.5`：宽松，保留更多词，语料膨胀
- `threshold=1`：平衡，保留约 90% 文档
- `threshold=2`：严格，保留约 60% 文档

**副作用**：多义词（如"苹果"）的低频含义可能被压制，因为其 TF 在对应文档中不够高。

## 3. 词向量 + KMeans 初始化

用词向量聚类为 LDA 提供更好的初始 topic 分配，替代随机初始化。

### 3.1 训练词向量

使用 [Wavec](https://github.com/Ismantic/Wavec)（CBOW + Hierarchical Softmax）：

```bash
wavec -dim 100 -window 5 -mincount 5 -threads 16 -iter 5 -sample 1e-3 \
    filtered.txt model.vec
```

### 3.2 KMeans 聚类

将词向量聚成 K 簇（K = topic 数），导出词-簇映射：

```bash
kmeans model.vec 128 50 0 --export word_topic_init.txt
```

输出格式为 `词 cluster_id`，一行一个词。

**注意事项**：
- 词向量词表应与 LDA 语料词表高度重叠（>90%），未覆盖的词回退随机分配
- 聚类分布应大致均匀，如果出现某个簇占 90%+ 的情况，说明词向量质量不够

### 3.3 覆盖率检查

```python
# 快速检查覆盖率
init_words = set(line.split()[0] for line in open("word_topic_init.txt"))
corpus_words = set(w for line in open("corpus.txt") for w in line.split())
overlap = init_words & corpus_words
print(f"覆盖率: {len(overlap)}/{len(corpus_words)} = {100*len(overlap)/len(corpus_words):.1f}%")
```

## 4. LDA 训练

### 4.1 基本训练

```bash
./build/semat corpus.txt 128 50 0.1 0.01 16
```

### 4.2 带 KMeans 初始化

```bash
./build/semat corpus.txt 128 150 0.1 0.01 16 --init word_topic_init.txt
```

### 4.3 参数建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| topics | 64~256 | 根据语料规模和期望粒度 |
| iterations | 100~200 | 观察 Perplexity 收敛情况 |
| alpha | 0.1 | 越小文档主题越集中 |
| beta | 0.01 | 越小主题词分布越尖锐 |
| num_cores | CPU 核数 | N-Queen 调度支持高并行 |

### 4.4 收敛判断

训练过程每 10 轮输出 Perplexity。当连续 10 轮降幅 < 5 时基本收敛：

```
Iteration  10: Perplexity = 969
Iteration  50: Perplexity = 840    # 降幅大，继续
Iteration 100: Perplexity = 799    # 降幅收窄
Iteration 130: Perplexity = 786    # 趋平
Iteration 150: Perplexity = 782    # 收敛
```

## 5. 结果查看

```bash
python3 scripts/show_topics.py semat.phi [topn=30]
```

## 6. 实验对比

以 THUCNews 语料（83 万篇新闻，128 topics）为例：

| 方案 | 50 轮 Perplexity | 150 轮 Perplexity | 主题质量 |
|------|-----------------|-------------------|---------|
| 原始语料 + 随机初始化 | 797 | - | 部分主题混杂 |
| TF-IDF 加权 + 随机初始化 | 840 | - | 主题更聚焦，多义词弱化 |
| TF-IDF 加权 + KMeans 初始化 | 840 | 782 | 主题纯净，收敛更快 |

**结论**：
- TF-IDF 加权降低噪声，使主题更聚焦
- KMeans 初始化加速收敛，前 10 轮即达到较好状态
- 两者结合 + 充分迭代效果最佳
