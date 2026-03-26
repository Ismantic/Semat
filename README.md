# Semat

Sparse Efficient Model Allocation Topic — 基于 SparseLDA + N-Queen 多线程调度的 LDA 主题模型实现。

## 特性

- **SparseLDA 采样**：将采样概率分解为 s/r/q 三个桶，降低采样复杂度
- **N-Queen 并行**：文档-词汇分块调度（modulo 分配），无锁多线程 Gibbs 采样
- **稀疏计数**：使用 `unordered_map` 存储计数矩阵，适合大规模稀疏数据
- **KMeans 初始化**：支持从词向量聚类结果初始化 topic 分配，加速收敛
- **Perplexity 监控**：训练过程中输出困惑度，评估模型收敛

详细的算法原理见 [LDA.md](LDA.md)，完整训练流程见 [docs/training-guide.md](docs/training-guide.md)。

## 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## 使用

```bash
./build/semat data.txt [topics] [iterations] [alpha] [beta] [num_cores] [--init file]
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `topics` | 128 | 主题数 K |
| `iterations` | 10 | 迭代次数 |
| `alpha` | 0.1 | 文档-主题先验 |
| `beta` | 0.01 | 主题-词语先验 |
| `num_cores` | CPU核数/2 | 线程数 |
| `--init` | 无 | 词-主题初始化文件（词 topic_id，一行一个） |

### 示例

基本训练：

```bash
./build/semat corpus.txt 128 150 0.1 0.01 16
```

带 KMeans 初始化：

```bash
./build/semat corpus.txt 128 150 0.1 0.01 16 --init word_topic_init.txt
```

### 输入文件格式

每行一个文档，词语以空格分隔（词袋格式，需预先分词）：

```
手机 苹果 软件 数据 算法 网络 程序
足球 比赛 球员 进球 教练 联赛 冠军
苹果 牛肉 餐厅 味道 烹饪 食材 美味
```

### 输出文件

训练完成后生成三个文件：

- `semat.vocab` — 词汇表
- `semat.phi` — 主题-词语分布（每个主题的 top 词语及概率）
- `semat.theta` — 文档-主题分布

## 工具

### TF-IDF 语料加权

```bash
python3 scripts/tfidf_reweight.py <input> <output> [--threshold 1]
```

对语料进行 `log10(TF) * log10(N/DF)` 加权，压制高频通用词，突出有区分度的词。

### 主题查看

```bash
python3 scripts/show_topics.py semat.phi [topn=30]
```

## 训练效果

以 THUCNews 语料（83 万篇新闻）训练 128 topics，TF-IDF 加权 + KMeans 初始化，150 轮迭代，Perplexity 782。部分主题示例：

| 主题 | Top 词 |
|------|--------|
| 大宗商品 | 黄金 大豆 原油 棉花 玉米 库存 石油 粮食 钢材 铁矿石 |
| NBA | 姚明 火箭 科比 篮网 湖人 詹姆斯 麦蒂 易建联 热火 韦德 |
| 欧洲足球 | 米兰 皇马 巴萨 曼联 梅西 切尔西 国米 阿森纳 利物浦 鲁尼 |
| 刑事案件 | 警方 民警 犯罪 嫌疑人 警察 诈骗 监狱 毒品 传销 团伙 |
| 互联网 | 苹果 谷歌 广告 百度 团购 微软 阿里 搜索 雅虎 腾讯 |
| 银行金融 | 银行 债券 投资 贷款 理财 评级 存款 融资 信用 信托 |
| 食品饮料 | 蔬菜 啤酒 奶粉 水果 饮料 咖啡 可乐 果汁 牛奶 鸡蛋 |
| 家居建材 | 家具 家居 陶瓷 卫浴 橱柜 红木 建材 木门 衣柜 经销商 |
| 国际冲突 | 朝鲜 伊朗 利比亚 阿富汗 以色列 伊拉克 卡扎菲 北约 塔利班 埃及 |
| 自然科学 | 太阳 地球 火星 海洋 科学家 火山 月球 恒星 行星 宇宙 |
| 电影影视 | 电影 票房 影片 电视剧 三国 观众 动漫 影视 红楼梦 收视 |
| 健康养生 | 减肥 健康 食物 睡眠 脂肪 按摩 热量 营养 饮食 皮肤 |
| 考试教育 | 考试 考研 考生 复习 英语 数学 阅读 题目 雅思 大纲 |
| 游戏 | 游戏 玩家 网游 盛大 网易 魔兽 巨人 暴雪 征途 竞技 |

完整 128 个主题见 [data/semat.phi](data/semat.phi)。

## License

MIT
