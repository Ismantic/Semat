# Semat

Sparse Efficient Model Allocation Topic — 基于 SparseLDA + N-Queen 多线程调度的 LDA 主题模型实现。

## 特性

- **SparseLDA 采样**：将采样概率分解为 s/r/q 三个桶，降低采样复杂度
- **N-Queen 并行**：文档-词汇分块调度，无锁多线程 Gibbs 采样
- **稀疏计数**：使用 `unordered_map` 存储计数矩阵，适合大规模稀疏数据
- **Perplexity 监控**：训练过程中输出困惑度，评估模型收敛

详细的算法原理见 [LDA.md](LDA.md)。

## 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## 使用

```bash
./build/semat data.txt [topics] [iterations] [alpha] [beta] [num_cores]
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `topics` | 128 | 主题数 K |
| `iterations` | 10 | 迭代次数 |
| `alpha` | 0.1 | 文档-主题先验 |
| `beta` | 0.01 | 主题-词语先验 |
| `num_cores` | CPU核数/2 | 线程数 |

输入文件每行一个文档，词语以空格分隔。

### 输出文件

训练完成后生成三个文件：

- `semat.vocab` — 词汇表
- `semat.phi` — 主题-词语分布（每个主题的 top 词语及概率）
- `semat.theta` — 文档-主题分布

## License

MIT
