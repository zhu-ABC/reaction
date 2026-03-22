# RAMER 完整代码开发需求文档

## 一、项目总体信息

### 1.1 项目名称
**RAMER — Retrieval-Augmented Missing-modality Emotion Recognition**  
（检索增强的缺失模态情绪识别）

### 1.2 核心任务
构建一个面向缺失模态场景的多模态情绪识别系统，输入模态包括：
- 文本（Text）
- 语音（Audio）
- 视觉（Visual）

系统目标是在任意模态缺失的条件下：
1. 基于可见模态先形成初步情绪判断；
2. 从情绪原型记忆库中检索与当前样本最相关的原型；
3. 对缺失模态进行补偿；
4. 通过选择性门控机制决定是否注入检索结果；
5. 最终完成情绪类别预测。

### 1.3 技术栈要求
```text
Python >= 3.9
PyTorch >= 2.0
transformers >= 4.30      # HuggingFace，用于 BERT / Wav2Vec2
scikit-learn >= 1.2       # KMeans、评估指标
numpy >= 1.24
librosa >= 0.10           # 语音处理
opensmile >= 2.4          # 可选，eGeMAPS 特征提取
pandas >= 2.0
pyyaml >= 6.0             # 配置文件解析
tensorboard >= 2.13       # 训练日志
matplotlib >= 3.7         # 可视化
seaborn >= 0.12           # 可视化
tqdm >= 4.65              # 进度条
```

### 1.4 完整文件结构
```text
RAMER/
├── configs/
│   ├── default.yaml                # 默认超参数配置
│   ├── iemocap.yaml                # IEMOCAP 专用配置
│   ├── mosi.yaml                   # MOSI 专用配置
│   └── mosei.yaml                  # MOSEI 专用配置
│
├── data/
│   ├── __init__.py
│   ├── preprocess_iemocap.py       # IEMOCAP 数据预处理脚本
│   ├── preprocess_mosi.py          # MOSI 数据预处理脚本
│   ├── preprocess_mosei.py         # MOSEI 数据预处理脚本
│   ├── dataset.py                  # PyTorch Dataset 类
│   ├── dataloader.py               # DataLoader 工厂 + 缺失采样器
│   └── missing_patterns.py         # 缺失模式定义与采样
│
├── models/
│   ├── __init__.py
│   ├── text_encoder.py             # 文本编码器
│   ├── audio_encoder.py            # 语音编码器
│   ├── visual_encoder.py           # 视觉编码器
│   ├── cross_relation_encoder.py   # 跨模态关系编码器
│   ├── prototype_bank.py           # 情绪原型记忆库
│   ├── observed_pool.py            # 可见模态自适应池化
│   ├── preliminary_predictor.py    # 初步情绪预测器
│   ├── retriever.py                # 情绪引导原型检索模块
│   ├── selective_gate.py           # 选择性注入门控
│   ├── fusion.py                   # 跨模态融合 + 分类器
│   └── ramer.py                    # 总模型封装 (RetrievalAugmentedMER)
│
├── losses/
│   ├── __init__.py
│   └── ramer_loss.py               # 全部损失函数
│
├── trainers/
│   ├── __init__.py
│   ├── stage1_trainer.py           # 阶段一训练器（完整数据预训练）
│   ├── stage2_trainer.py           # 阶段二训练器（检索模块训练）
│   └── bank_builder.py             # 原型库构建器
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py                # 评估主模块
│   ├── metrics.py                  # WAR/UAR/WF1 等指标计算
│   └── missing_protocols.py        # 标准化缺失评估协议
│
├── analysis/
│   ├── __init__.py
│   ├── gate_analysis.py            # 门控值分析与可视化
│   ├── retrieval_analysis.py       # 检索结果分析与可视化
│   ├── tsne_visualization.py       # t-SNE 特征空间可视化
│   ├── ablation.py                 # 消融实验自动化脚本
│   └── case_study.py               # 案例分析脚本
│
├── utils/
│   ├── __init__.py
│   ├── config.py                   # 配置加载与合并
│   ├── logger.py                   # 日志工具
│   ├── seed.py                     # 随机种子设置
│   ├── checkpoint.py               # 模型保存与加载
│   └── misc.py                     # 杂项工具函数
│
├── scripts/
│   ├── run_preprocess.sh           # 数据预处理脚本
│   ├── run_train.sh                # 训练启动脚本
│   ├── run_eval.sh                 # 评估脚本
│   └── run_ablation.sh             # 消融实验脚本
│
├── train.py                        # 训练主入口
├── evaluate.py                     # 评估主入口
├── analyze.py                      # 分析主入口
├── requirements.txt                # 依赖包
└── README.md                       # 项目说明 / 开发需求文档
```

---

## 二、配置文件需求

### 2.1 `configs/default.yaml`
```yaml
# ===== 数据 =====
data:
  dataset: "iemocap"              # iemocap / mosi / mosei
  data_dir: "./processed_data"     # 预处理后的数据目录
  num_classes: 4                   # IEMOCAP=4, MOSI=2, MOSEI=2
  text_max_len: 50                 # 文本最大 token 数
  audio_max_len: 100               # 语音最大帧数
  visual_max_len: 60               # 视觉最大帧数
  text_feat_dim: 768               # BERT 输出维度
  audio_feat_dim: 768              # wav2vec2 输出维度（或 88 用 eGeMAPS）
  visual_feat_dim: 35              # OpenFace 输出维度（或 512 用 ResNet）
  num_workers: 4                   # DataLoader workers

# ===== 模型 =====
model:
  d: 256                           # 统一特征维度
  num_heads: 4                     # 注意力头数
  ffn_dim: 512                     # Transformer FFN 维度
  dropout: 0.1                     # Dropout 率
  encoder_layers: 1                # 每个模态编码器的 Transformer 层数
  fusion_layers: 2                 # 跨模态融合的 Transformer 层数
  freeze_bert: true                # 是否冻结 BERT
  bert_model: "bert-base-uncased"  # BERT 模型名

# ===== 原型库 =====
bank:
  K: 50                            # 每类每模态的原型数
  top_k: 5                         # 检索时取 Top-k
  tau: 0.07                        # 检索 softmax 温度
  kmeans_n_init: 10                # KMeans 初始化次数
  kmeans_random_state: 42          # KMeans 随机种子

# ===== 训练 =====
training:
  seed: 42
  stage1:
    epochs: 30
    lr: 1.0e-4
    weight_decay: 1.0e-4
    batch_size: 32
    grad_clip: 1.0
    patience: 10
    scheduler: "cosine"           # cosine / step / none
  stage2:
    epochs: 50
    lr: 5.0e-5
    weight_decay: 1.0e-4
    batch_size: 32
    grad_clip: 1.0
    patience: 15
    scheduler: "cosine"
    freeze_encoders: true

# ===== 损失权重 =====
loss:
  lambda_retrieval: 0.5
  lambda_gate: 0.3
  lambda_consistency: 0.3
  lambda_pre_cls: 0.2

# ===== 缺失模拟 =====
missing:
  full_prob: 0.15
  single_miss_prob: 0.25
  double_miss_prob: 0.25
  text_miss_prob: 0.15
  text_only_prob: 0.10
  av_miss_prob: 0.10

# ===== 评估 =====
evaluation:
  protocols:
    - "full"
    - "miss_t"
    - "miss_a"
    - "miss_v"
    - "miss_ta"
    - "miss_tv"
    - "miss_av"
  n_folds: 5

# ===== 日志与保存 =====
output:
  save_dir: "./checkpoints"
  log_dir: "./logs"
  save_best: true
  save_every: 10
```

### 2.2 其他配置文件要求
- `configs/iemocap.yaml`：覆盖 IEMOCAP 数据路径、类别数、5 折评估、预处理参数。
- `configs/mosi.yaml`：覆盖 MOSI 的类别数、标准划分、二分类或可选回归设置。
- `configs/mosei.yaml`：覆盖 MOSEI 的类别数、标准划分、规模与 batch 策略。
- 配置加载必须支持：
  - 默认配置 + 数据集专用配置合并；
  - 命令行参数二次覆盖；
  - 缺失字段自动回退默认值并打印 warning。

---

## 三、数据预处理需求

### 3.1 `data/preprocess_iemocap.py`
**输入：** 原始 IEMOCAP 数据集路径。  
**输出：** 处理后的 `.pt` 文件，每个样本一个文件，外加 `metadata.json`。

#### 功能要求
1. 遍历 IEMOCAP 的 5 个 session。
2. 仅保留 4 类情绪：
   - happy（合并 excited）
   - sad
   - angry
   - neutral
3. 对每个话语依次执行：
   - **文本处理**
     - 读取转录文本；
     - 使用 `bert-base-uncased` tokenizer 编码；
     - `padding/truncation` 到 `max_len=50`；
     - 输出 `input_ids [50]` 与 `attention_mask [50]`。
   - **语音处理**
     - 使用 `librosa` 以 `sr=16000` 加载音频；
     - 使用预训练 `wav2vec2-base` 提取帧级特征；
     - 输出 `[T, 768]`；
     - `padding/truncation` 到 `max_len=100`；
     - 输出 `length_mask [100]`。
   - **视觉处理**
     - 使用 OpenFace 2.0 提取面部动作单元与头部/视线信息；
     - 维度约 35：17 AU intensity + 6 AU occurrence + 头部姿态 3 + 凝视 4 + 置信度 1 + 其他有效字段；
     - 输出 `[T, 35]`；
     - `padding/truncation` 到 `max_len=60`；
     - 输出 `length_mask [60]`。
4. 每个样本保存为：
```python
{
    'sample_id': str,
    'text_ids': LongTensor,       # [50]
    'text_mask': LongTensor,      # [50]
    'audio': FloatTensor,         # [100, 768]
    'audio_mask': LongTensor,     # [100]
    'visual': FloatTensor,        # [60, 35]
    'visual_mask': LongTensor,    # [60]
    'label': int,                 # 0=happy, 1=sad, 2=angry, 3=neutral
    'speaker_id': str,
    'session_id': int,            # 1~5
    'text_raw': str,
}
```
5. 同时保存 `metadata.json`：
```json
{
  "total_samples": 0,
  "class_distribution": {"0": 0, "1": 0, "2": 0, "3": 0},
  "session_samples": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
  "class_names": ["happy", "sad", "angry", "neutral"]
}
```

#### 注意事项
- wav2vec2 提取优先在 GPU 上执行。
- OpenFace 若不可用，必须支持直接读取预提取 `.csv` 特征。
- 缺失片段统一零填充，并在 mask 中置 0。
- 使用 `tqdm` 显示处理进度。

### 3.2 `data/preprocess_mosi.py` 与 `data/preprocess_mosei.py`
#### 功能要求
1. 使用 `CMU-MultimodalSDK` 加载对齐后数据，或兼容常见 MMSA 风格预处理格式。
2. 文本、语音、视觉的处理方式与 IEMOCAP 保持一致。
3. 标签处理支持：
   - 二分类：`sentiment > 0 => positive(1)`，否则 `negative(0)`；
   - 七分类（可选）；
   - 回归（可选）。
4. 采用标准 `train/valid/test` 划分。
5. 输出格式与 IEMOCAP 一致。

#### 数据规模参考
- MOSI：约 2199 样本。
- MOSEI：约 22856 样本。

### 3.3 `data/dataset.py`
类名：`MultimodalEmotionDataset`

#### 构造参数
- `data_dir: str`
- `split: str`，`train / val / test`
- `session_ids: list[int]`，IEMOCAP 专用
- `dataset_name: str`，`iemocap / mosi / mosei`

#### `__getitem__` 返回格式
```python
{
    'sample_id': str,
    'text_ids': LongTensor,     # [50]
    'text_mask': LongTensor,    # [50]
    'audio': FloatTensor,       # [100, 768]
    'audio_mask': LongTensor,   # [100]
    'visual': FloatTensor,      # [60, 35]
    'visual_mask': LongTensor,  # [60]
    'label': LongTensor,        # scalar
}
```

#### `collate_fn`
- 标准 batch 拼接；
- 输出各字段 shape 为 `[B, ...]`；
- 必须保留 `sample_id` 列表供分析与案例导出使用。

#### 额外方法
- `get_class_weights() -> FloatTensor [C]`
- `get_sample_ids_by_class(class_id) -> list[str]`

### 3.4 `data/missing_patterns.py`
#### 预定义缺失模式
```python
MISSING_PATTERNS = {
    'full':    {'t': 1, 'a': 1, 'v': 1},
    'miss_t':  {'t': 0, 'a': 1, 'v': 1},
    'miss_a':  {'t': 1, 'a': 0, 'v': 1},
    'miss_v':  {'t': 1, 'a': 1, 'v': 0},
    'miss_ta': {'t': 0, 'a': 0, 'v': 1},
    'miss_tv': {'t': 0, 'a': 1, 'v': 0},
    'miss_av': {'t': 1, 'a': 0, 'v': 0},
}
```

#### 类 `MissingPatternSampler`
- 从 `config.missing` 读取概率。
- `sample() -> dict`
- `sample_batch(batch_size) -> dict`
- 要保证**同一 batch 使用同一种缺失模式**。

#### 函数 `apply_missing_mask(batch, mask) -> batch`
- 返回深拷贝后的 batch；
- 模态缺失时执行：
  - `t=0`：`text_ids = 0`，`text_mask = 0`
  - `a=0`：`audio = 0.0`，`audio_mask = 0`
  - `v=0`：`visual = 0.0`，`visual_mask = 0`

---

## 四、模型各模块详细需求

### 4.1 `models/text_encoder.py`
类名：`TextEncoder`，继承 `nn.Module`

#### 构造参数
- `d: int = 256`
- `bert_model: str = 'bert-base-uncased'`
- `freeze_bert: bool = True`
- `dropout: float = 0.1`
- `num_heads: int = 4`
- `ffn_dim: int = 512`

#### 结构
1. `self.bert = BertModel.from_pretrained(bert_model)`
2. 冻结 BERT（若 `freeze_bert=True`）
3. `self.proj = Sequential(Linear(768, d), LayerNorm(d), GELU(), Dropout(dropout))`
4. `self.temporal = TransformerEncoder(...)`

#### `forward(input_ids, attention_mask)`
- 输入：`[B, 50]`
- 输出：
  - `H_t: [B, 50, d]`
  - `h_t: [B, d]`
- 使用带 mask 的均值池化。

### 4.2 `models/audio_encoder.py`
类名：`AudioEncoder`

#### 输入输出
- 输入：
  - `x_a: [B, 100, input_dim]`
  - `length_mask: [B, 100]`
- 输出：
  - `H_a: [B, 100, d]`
  - `h_a: [B, d]`

#### 结构
- 线性投影
- 一维卷积局部建模 + 残差
- TransformerEncoder 时序建模
- 带 mask 均值池化

### 4.3 `models/visual_encoder.py`
类名：`VisualEncoder`
- 结构与 `AudioEncoder` 一致；
- `input_dim = 35`；
- 输出：
  - `H_v: [B, 60, d]`
  - `h_v: [B, d]`

### 4.4 `models/cross_relation_encoder.py`
类名：`CrossRelationEncoder`

#### 核心逻辑
- 逐元素构建：
  - `r_ta = h_t * h_a`
  - `r_tv = h_t * h_v`
  - `r_av = h_a * h_v`
- 拼接为 `[B, 3d]` 后用 MLP 融合为 `[B, d]`。

### 4.5 `models/prototype_bank.py`
类名：`PrototypeBank`

#### 核心属性
- `self.bank[class_id][modality] = Tensor[K, d]`
- 模态集合：`'t'`, `'a'`, `'v'`, `'cross'`
- `class_means`
- `global_mean`
- `is_built`

#### `build(...)`
1. 所有编码器切换 `eval()`；
2. 遍历训练集提取 `h_t`, `h_a`, `h_v`, `h_cross`；
3. 按类别收集；
4. 对每个类别-模态执行 KMeans：
   - `n_clusters = min(K, num_samples)`；
   - 若样本少于 2，直接均值重复；
   - 若聚类数不足 K，则重复填满；
5. 生成 `bank`、`class_means`、`global_mean`；
6. `self.is_built = True`；
7. 打印统计信息。

#### 其他接口
- `get_prototypes(modality, device)`
- `get_class_prototypes(class_id, modality, device)`
- `get_global_mean(modality, device)`
- `save(path)`
- `load(path)`

### 4.6 `models/observed_pool.py`
类名：`ObservedModalityPool`

#### 功能
通过一个可学习全局 query 对可见模态集合执行交叉注意力池化，输出统一的可见模态表示 `q_obs`。

#### 输出
- `q_obs: [B, d]`
- `attn_weights: [B, |obs|]`

### 4.7 `models/preliminary_predictor.py`
类名：`PreliminaryPredictor`

#### 输出
- `prob: [B, C]`
- `logit: [B, C]`

#### 作用
- 为检索提供情绪先验；
- 作为辅助监督分支训练。

### 4.8 `models/retriever.py`
类名：`EmotionGuidedRetriever`

#### 核心设计
1. 为每个缺失模态建立独立 query 投影；
2. 为跨模态关系建立 `cross_query_proj`；
3. 使用 `prob_pre` 作为情绪引导，对原型相似度进行类别加权；
4. 检索 `top_k` 原型并做软聚合；
5. 单模态原型结果与关系原型结果再做融合；
6. 输出检索特征及置信度相关统计量。

#### `retrieve_single_modality(...)` 输出
```python
{
    'retrieved': FloatTensor,      # [B, d]
    'max_sim': FloatTensor,        # [B]
    'label_entropy': FloatTensor,  # [B]
    'topk_labels': LongTensor,     # [B, k]
}
```

#### `forward(...)` 输出
```python
results = {
    missing_modality: {
        'retrieved': FloatTensor,      # [B, d]
        'max_sim': FloatTensor,        # [B]
        'label_entropy': FloatTensor,  # [B]
        'topk_labels': LongTensor,     # [B, k]
        'alpha': FloatTensor,          # [B]
    }
}
```

### 4.9 `models/selective_gate.py`
类名：`SelectiveGate`

#### 输入信号
- `max_sim`
- `label_entropy`
- `cosine_similarity(retrieved, q_obs)`
- `obs_ratio`
- `alpha`

#### 输出
- `h_compensated: [B, d]`
- `gate_value: [B]`

### 4.10 `models/fusion.py`
类名：`CrossModalFusion`

#### 设计要求
- 三个模态 token 加上：
  - 模态身份嵌入；
  - 来源嵌入（原始 / 检索补偿）。
- 使用 TransformerEncoder 融合；
- 再用可学习 query 做 attention pooling；
- 最后输出：
  - `logit: [B, C]`
  - `h_fused: [B, d]`

### 4.11 `models/ramer.py`
类名：`RAMER`（或 `RetrievalAugmentedMER`）

#### 初始化要求
从配置中读取参数，初始化：
- 文本编码器
- 音频编码器
- 视觉编码器
- 跨模态关系编码器
- 原型库
- 可见模态池化器
- 初步预测器
- 检索器
- 选择性门控
- 融合分类器

#### `encode_all(batch, mask)`
- 若模态可见，执行对应编码；
- 若不可见，直接返回 `None`；
- 返回：
  - `h_dict`
  - `H_dict`

#### `forward(batch, mask)` 分支逻辑
1. 解析 `obs_set` 与 `mis_set`；
2. 编码所有可见模态；
3. 三种场景：
   - **完整输入**：直接融合分类；
   - **全部缺失**：回退到 `global_mean` 原型，并打印 warning；
   - **部分缺失**：
     - `ObservedModalityPool`
     - `PreliminaryPredictor`
     - `EmotionGuidedRetriever`
     - `SelectiveGate`
     - `CrossModalFusion`
4. 返回：
```python
{
    'logit': FloatTensor,
    'h_fused': FloatTensor,
    'gate_values': dict[str, FloatTensor],
    'pre_logit': FloatTensor | None,
    'retrieved': dict[str, FloatTensor],
    'pre_prob': FloatTensor | None,
}
```

#### 其他接口
- `freeze_encoders()`
- `unfreeze_encoders()`
- `get_trainable_params()`
- `count_parameters()`

---

## 五、损失函数需求

### 5.1 `losses/ramer_loss.py`
类名：`RAMERLoss`

#### 损失组成
1. **分类损失 `L_cls`**
2. **检索质量损失 `L_retrieval`**
3. **门控校准损失 `L_gate`**
4. **完整-缺失一致性损失 `L_consistency`**
5. **初步预测辅助损失 `L_pre_cls`**

#### 总损失
```text
L_total = L_cls
        + λ1 * L_retrieval
        + λ2 * L_gate
        + λ3 * L_consistency
        + λ4 * L_pre_cls
```

#### `forward(...)` 输出
```python
{
    'total': FloatTensor,
    'cls': FloatTensor,
    'retrieval': FloatTensor,
    'gate': FloatTensor,
    'consistency': FloatTensor,
    'pre_cls': FloatTensor,
}
```

---

## 六、训练器需求

### 6.1 `trainers/stage1_trainer.py`
类名：`Stage1Trainer`

#### 训练目标
在**完整输入**上训练：
- 编码器
- 跨模态关系编码器
- 融合层
- 分类器

#### 训练流程
1. 使用 `AdamW`；
2. 只优化：`text_enc`, `audio_enc`, `visual_enc`, `cross_rel_enc`, `fusion`；
3. scheduler 默认使用 `CosineAnnealingLR`；
4. 每个 batch 使用 `mask={'t':1,'a':1,'v':1}`；
5. 损失为分类交叉熵；
6. 使用梯度裁剪；
7. 每轮后在验证集评估 `WAR/UAR/WF1`；
8. 以 `val_WAR` 为最优模型选择指标；
9. 支持早停；
10. 返回最佳模型路径。

### 6.2 `trainers/bank_builder.py`
类名：`BankBuilder`

#### 功能
1. 使用训练集构建原型库；
2. 保存到 `save_path`；
3. 打印：
   - 各类别样本数；
   - 各类别原型数；
   - 原型库大小（MB）；
4. 在训练集上计算原型检索类别一致率并输出：
   - `Prototype retrieval accuracy: xx.x%`

### 6.3 `trainers/stage2_trainer.py`
类名：`Stage2Trainer`

#### 训练目标
训练：
- 检索模块
- 门控模块
- 初步预测器
- 可见模态池化器
- 融合层
- 可选微调编码器

#### 训练流程
1. 若配置要求则冻结编码器；
2. 使用 `AdamW` 优化所有可训练参数；
3. 初始化 scheduler；
4. 初始化 `MissingPatternSampler`；
5. 初始化 `RAMERLoss`；
6. 每个 batch：
   - 采样统一缺失模式；
   - `torch.no_grad()` 下跑完整前向，得到 `output_full` 与 `h_real_dict`；
   - 再跑缺失前向，得到 `output_miss`；
   - 计算组合损失并反向传播；
   - 梯度裁剪；
   - 写入 TensorBoard。
7. 每轮后对所有协议评估；
8. 以缺失协议平均 `avg_war` 作为最优模型指标；
9. 支持早停。

#### `evaluate(loader, miss_type)` 输出
```python
{
    'WAR': float,
    'UAR': float,
    'WF1': float,
    'gate_stats': {
        't': {'mean': float, 'var': float, 'reject_rate': float},
        'a': {'mean': float, 'var': float, 'reject_rate': float},
        'v': {'mean': float, 'var': float, 'reject_rate': float},
    }
}
```

---

## 七、评估模块需求

### 7.1 `evaluation/metrics.py`
函数：`compute_metrics(y_true, y_pred, num_classes)`

#### 输出
```python
{
    'WAR': float,
    'UAR': float,
    'WF1': float,
    'per_class_f1': dict[int, float],
    'confusion_matrix': np.ndarray,
}
```

### 7.2 `evaluation/missing_protocols.py`
类名：`MissingProtocolEvaluator`

#### `evaluate_all_protocols()`
依次评估：
- `full`
- `miss_t`
- `miss_a`
- `miss_v`
- `miss_ta`
- `miss_tv`
- `miss_av`

#### 返回格式
```python
{
    'full': {...},
    'miss_t': {...},
    'miss_a': {...},
    'miss_v': {...},
    'miss_ta': {...},
    'miss_tv': {...},
    'miss_av': {...},
    'avg_miss': {'WAR': x, 'UAR': x, 'WF1': x},
}
```

#### `evaluate_random_missing(missing_rates)`
- 输入示例：`[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]`
- 对每个样本按 rate 独立随机丢弃模态；
- 输出每个 rate 下的指标结果。

#### 其他方法
- `print_results_table(results)`：打印统一格式表格；
- `save_results(results, path)`：保存 JSON。

---

## 八、分析与可视化需求

### 8.1 `analysis/gate_analysis.py`
函数：`analyze_gate_values(model, test_loader, device, save_dir)`

#### 输出内容
1. `gate_distribution.pdf`
2. `gate_vs_similarity.pdf`
3. `gate_vs_accuracy.pdf`
4. `rejection_rate.pdf`

### 8.2 `analysis/retrieval_analysis.py`
函数：`analyze_retrieval(model, test_loader, device, save_dir)`

#### 输出内容
- 检索类别一致率统计；
- `retrieval_confusion.pdf`；
- `alpha_distribution.pdf`；
- `retrieval_cases.txt`（5 个成功案例 + 5 个失败案例）。

### 8.3 `analysis/tsne_visualization.py`
函数：`visualize_tsne(model, test_loader, device, save_dir)`

#### 输出内容
- `tsne_full.pdf`
- `tsne_missing.pdf`
- `tsne_before_after_retrieval.pdf`
- `tsne_prototypes.pdf`

### 8.4 `analysis/ablation.py`
函数：`run_ablation(base_config, device)`

#### 需要支持的消融项
1. No Retrieval
2. No Gate (Always Inject)
3. No Emotion Guidance
4. No Relation Prototype
5. No Consistency Loss
6. No Pre-classifier
7. `K=10 / 100 / 200`
8. `top_k=1 / 3 / 10`
9. Generation Baseline

#### 输出
- `ablation_results.csv`
- `ablation_results.pdf`
- 控制台表格汇总

---

## 九、主入口脚本需求

### 9.1 `train.py`
#### 命令行参数
```text
--config
--dataset
--fold
--stage
--resume
--gpu
--seed
```

#### 运行流程
1. 加载配置；
2. 合并命令行覆盖项；
3. 设置随机种子；
4. 设置 device；
5. 若 IEMOCAP 且 `fold == -1`，执行 5 折训练；
6. 每折执行：
   - 构建 DataLoader；
   - 初始化模型；
   - 阶段一训练；
   - 构建原型库；
   - 阶段二训练；
   - 测试集全协议评估；
   - 保存结果；
7. 若是 5 折，汇总平均值与标准差并打印最终表格。

### 9.2 `evaluate.py`
#### 命令行参数
```text
--config
--checkpoint
--bank_path
--dataset
--fold
--protocols
--random_rates
--gpu
--output
```

#### 流程
1. 加载配置、模型、checkpoint、bank；
2. 构建测试集 DataLoader；
3. 运行固定协议评估；
4. 可选运行随机缺失率评估；
5. 打印表格并保存结果。

### 9.3 `analyze.py`
#### 命令行参数
```text
--config
--checkpoint
--bank_path
--analysis
--gpu
--save_dir
```

#### 流程
根据 `--analysis` 调用：
- `gate`
- `retrieval`
- `tsne`
- `ablation`
- `all`

---

## 十、工具函数需求

### 10.1 `utils/config.py`
#### 必须提供
- `load_config(yaml_path) -> dict`
- `merge_config(base_config, override_dict) -> dict`
- `Config`：支持点号访问

### 10.2 `utils/seed.py`
#### 必须提供
- `set_seed(seed)`
- 同步设置 `random / numpy / torch / torch.cuda`
- 设置：
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`

### 10.3 `utils/checkpoint.py`
#### 必须提供
- `save_checkpoint(...)`
- `load_checkpoint(...)`
- `save_best_model(...)`
- `load_model(...)`

### 10.4 `utils/logger.py`
类名：`Logger`

#### 能力要求
- 文件日志 + 控制台日志 + TensorBoard；
- `info(msg)`；
- `scalar(tag, value, step)`；
- `table(headers, rows)`；
- 输出格式与训练日志示例保持一致。

---

## 十一、对比方法实现需求

在 `models/baselines/` 下实现：
```text
models/baselines/
├── zero_imputation.py
├── mean_imputation.py
├── generation_baseline.py
├── mmin.py
└── __init__.py
```

### 通用要求
- 每个 baseline 提供统一接口：
```python
forward(batch, mask) -> {
    'logit': ...,
    'h_fused': ...,
}
```

### `generation_baseline.py`
#### 类名
`GenerationBaseline`

#### 设计要求
- 编码器沿用 RAMER 同构编码器；
- 对每个缺失模态建立生成器：
```python
Sequential(
    Linear(d * |obs|, d * 2),
    GELU(),
    Linear(d * 2, d)
)
```
- 融合器沿用 `CrossModalFusion`；
- 损失包含：
  - `L_cls`
  - `L_reconstruction`

#### 目标
作为“检索优于生成”的对照基线。

---

## 十二、需要特别注意的实现细节

### 12.1 数据泄漏防护
- 原型库只允许使用训练集构建；
- 每一折 IEMOCAP 都要单独构建原型库；
- 检索时只能从原型库搜索，不能从当前 batch 搜索。

### 12.2 Batch 内 mask 统一
- 同一 batch 全部样本共享同一缺失模式；
- 缺失模态的编码器直接跳过，不做前向传播。

### 12.3 梯度流控制
- 阶段一：编码器 + 融合层 + 分类器参与训练；
- 阶段二：编码器可冻结；
- 原型库始终不参与梯度；
- `h_real_dict`、`output_full` 全部 `detach()`。

### 12.4 数值稳定性
- `F.normalize` 后再算余弦相似度；
- 所有 `log` 前加 `1e-8`；
- 除法前 `clamp(min=1e-8)`；
- `tau >= 0.01`。

### 12.5 显存优化
- 阶段二的完整前向使用 `torch.no_grad()`；
- 原型库存储时全部 `detach()`；
- BERT 冻结后避免梯度缓存；
- OOM 时优先减小 `batch_size`。

### 12.6 可复现性
- 所有随机过程受 seed 控制；
- KMeans 固定 `random_state`；
- DataLoader 固定 generator seed。

### 12.7 设备管理
- 模型、数据、原型库必须统一 device；
- 新建 tensor 时显式指定 `device`。

---

## 十三、预期输出格式

### 13.1 训练日志格式
```text
[2024-xx-xx xx:xx:xx] ===== Stage 1 Training =====
[Epoch 01/30] Train Loss: 1.2345 | Val WAR: 62.34 | Val UAR: 60.12 | Val WF1: 61.23
[Epoch 02/30] Train Loss: 0.9876 | Val WAR: 65.78 | Val UAR: 63.45 | Val WF1: 64.56
...
[Epoch 28/30] Train Loss: 0.3210 | Val WAR: 78.90 | Val UAR: 77.12 | Val WF1: 78.01 ★ Best
[Stage 1] Best model saved to checkpoints/stage1_best_fold1.pt

===== Building Prototype Bank =====
Class 0 (happy):  782 samples → 50 prototypes
Class 1 (sad):    741 samples → 50 prototypes
Class 2 (angry):  803 samples → 50 prototypes
Class 3 (neutral): 698 samples → 50 prototypes
Prototype retrieval accuracy: 87.3%
Bank saved to checkpoints/bank_fold1.pt

===== Stage 2 Training =====
[Epoch 01/50] L_total: 1.567 | L_cls: 0.892 | L_ret: 0.342 | L_gate: 0.201 | L_cons: 0.098 | L_pre: 0.034
┌──────────┬───────┬───────┬───────┐
│ Protocol │  WAR  │  UAR  │  WF1  │
├──────────┼───────┼───────┼───────┤
│ Full     │ 78.90 │ 77.12 │ 78.01 │
│ Miss-T   │ 61.23 │ 59.45 │ 60.34 │
│ Miss-A   │ 72.34 │ 70.56 │ 71.45 │
│ Miss-V   │ 74.56 │ 72.78 │ 73.67 │
│ Miss-TA  │ 55.67 │ 53.89 │ 54.78 │
│ Miss-TV  │ 58.90 │ 57.12 │ 58.01 │
│ Miss-AV  │ 69.01 │ 67.23 │ 68.12 │
│ Avg Miss │ 65.29 │ 63.51 │ 64.40 │
└──────────┴───────┴───────┴───────┘
```

### 13.2 最终结果表格格式
```text
===== IEMOCAP 5-Fold Average Results =====
┌──────────┬───────────────┬───────────────┬───────────────┐
│ Protocol │ WAR (±std)    │ UAR (±std)    │ WF1 (±std)    │
├──────────┼───────────────┼───────────────┼───────────────┤
│ Full     │ 79.12 (±1.23) │ 77.34 (±1.45) │ 78.23 (±1.34) │
│ Miss-T   │ 63.45 (±2.12) │ 61.67 (±2.34) │ 62.56 (±2.23) │
│ Miss-A   │ 73.56 (±1.89) │ 71.78 (±2.01) │ 72.67 (±1.95) │
│ Miss-V   │ 75.67 (±1.67) │ 73.89 (±1.89) │ 74.78 (±1.78) │
│ Miss-TA  │ 57.89 (±2.45) │ 56.11 (±2.67) │ 57.00 (±2.56) │
│ Miss-TV  │ 60.12 (±2.34) │ 58.34 (±2.56) │ 59.23 (±2.45) │
│ Miss-AV  │ 70.23 (±2.01) │ 68.45 (±2.23) │ 69.34 (±2.12) │
├──────────┼───────────────┼───────────────┼───────────────┤
│ Avg Miss │ 66.82 (±1.78) │ 65.04 (±2.00) │ 65.93 (±1.89) │
└──────────┴───────────────┴───────────────┴───────────────┘
```

---

## 十四、附加要求

### 14.1 代码风格
- 所有函数、类必须有 docstring；
- 关键步骤必须有行内注释；
- 复杂 tensor shape 变化要显式标注；
- 命名清晰，避免无意义单字母变量。

### 14.2 错误处理
- 原型库未构建就检索：抛出 `RuntimeError`；
- 全模态缺失：fallback 到全局均值并打印 warning；
- 配置字段缺失：回退默认值并打印 warning；
- GPU OOM：捕获异常并提示降低 `batch_size`。

### 14.3 测试要求
每个核心模块至少提供一个 smoke test，覆盖：
- `forward` 输入输出 shape；
- 阶段一梯度传播正确；
- 阶段二检索模块梯度传播正确；
- 冻结后编码器参数不更新；
- 原型库构建与加载行为正确。

### 14.4 README 必须包含内容
- 项目简介；
- 环境安装步骤；
- 数据下载与预处理说明；
- 训练命令示例；
- 评估命令示例；
- 结果复现说明；
- 论文引用格式。

---

## 十五、建议的开发里程碑

### Milestone 1：基础工程搭建
- 建立完整目录结构；
- 编写配置系统、日志系统、checkpoint 系统；
- 搭建 Dataset 与 DataLoader；
- 补齐 requirements 与脚本入口。

### Milestone 2：完整模态主干训练
- 完成三模态编码器；
- 完成跨模态关系编码器与融合分类器；
- 实现 Stage 1 训练与验证；
- 确保完整输入基线可复现。

### Milestone 3：原型库与检索补偿
- 完成 PrototypeBank；
- 实现 Observed Pool、Preliminary Predictor、Retriever、Selective Gate；
- 完成 Stage 2 训练与多协议评估。

### Milestone 4：分析与论文配套实验
- 完成固定缺失与随机缺失评估；
- 完成门控、检索、t-SNE 分析；
- 跑完整消融实验与 baseline 对比；
- 导出论文结果表格与可视化图片。

---

## 十六、验收标准

项目交付时，至少应满足以下验收条件：
1. 能够在 IEMOCAP 上完成 5 折训练、构建原型库、测试全协议；
2. 能够在 MOSI/MOSEI 上以标准划分完成训练与评估；
3. 所有缺失协议都能正常推理，无 shape / device / mask 错误；
4. 原型库严格无数据泄漏；
5. 所有核心模块具备 smoke test；
6. 日志、checkpoint、结果表格、可视化文件可自动保存；
7. README 能支持他人在新环境中复现实验流程。
