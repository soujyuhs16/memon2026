# 数据管线文档 / Data Pipeline Documentation

## 概述 / Overview

本文档描述如何使用新的数据管线构建检测器专用数据集。数据管线支持从 ZHateBench 和 HuggingFace 数据集构建四个检测器（D1-D4）的训练/验证/测试数据。

This document describes how to use the new data pipeline to build detector-specific datasets. The pipeline supports building train/dev/test data for four detectors (D1-D4) from ZHateBench and HuggingFace datasets.

## 数据来源 / Data Sources

### ZHateBench

**引用 / Citation:**
```
DOI: 10.5281/zenodo.16812052
URL: https://doi.org/10.5281/zenodo.16812052
```

**⚠️ 重要提示 / Important Notice:**
- 数据包含敏感内容，仅供研究使用
- Contains sensitive content, research use only
- 不得用于商业目的 / Not for commercial use
- 仅在学校内保存，不公开 / Only stored within school, not public

**数据集组成 / Dataset Composition:**

ZHateBench 包含 6 个类别的 CSV 文件：

1. **SexHarmSet.csv** - 涉黄内容 (Pornographic content) → D1 检测器
2. **AbuseSet.csv** - 辱骂内容 (Abusive content) → D2 检测器
3. **Bias_region.csv** - 地域歧视 (Regional bias) → D3 检测器
4. **BiasSet_genden.csv** - 性别歧视 (Gender bias) → D3 检测器
5. **Bias_race.csv** - 种族歧视 (Racial bias) → D3 检测器
6. **Bias_occupation.csv** - 职业歧视 (Occupational bias) → D3 检测器

**CSV 列结构 / CSV Column Structure:**
- `Keyword`: 关键词（用于可解释性）
- `Type`: `Harmful` 或 `Safe`
- `Sentence`: 文本内容

### Spam 数据集 / Spam Dataset

**来源 / Source:**
```
HuggingFace: reatiny/chinese-spam-10000
URL: https://huggingface.co/datasets/reatiny/chinese-spam-10000
```

用于 D4 检测器（广告诈骗/引流检测）。

## 目录结构 / Directory Structure

```
memon2026/
├── data_raw/
│   └── zhatebench/          # 存放 ZHateBench 6 个 CSV 文件
│       ├── SexHarmSet.csv
│       ├── AbuseSet.csv
│       ├── Bias_region.csv
│       ├── BiasSet_genden.csv
│       ├── Bias_race.csv
│       └── Bias_occupation.csv
│
├── data_processed/          # 生成的处理后数据
│   ├── d1_porn/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   ├── d2_abuse/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   ├── d3_bias/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   ├── d4_spam/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   └── manifest.json        # 构建元数据
│
├── scripts/
│   └── build_detectors_datasets.py  # 数据构建脚本
│
└── src/
    ├── data/
    │   ├── zhatebench.py    # ZHateBench 加载工具
    │   └── build_splits.py  # 数据切分工具
    └── train.py             # 训练脚本（已更新）
```

## 步骤 1: 准备 ZHateBench 数据 / Step 1: Prepare ZHateBench Data

1. **获取 ZHateBench 数据集**（通过学校/研究机构渠道）

2. **将 6 个 CSV 文件复制到 `data_raw/zhatebench/` 目录：**

```bash
mkdir -p data_raw/zhatebench

# 复制文件（根据实际路径调整）
cp /path/to/SexHarmSet.csv data_raw/zhatebench/
cp /path/to/AbuseSet.csv data_raw/zhatebench/
cp /path/to/Bias_region.csv data_raw/zhatebench/
cp /path/to/BiasSet_genden.csv data_raw/zhatebench/
cp /path/to/Bias_race.csv data_raw/zhatebench/
cp /path/to/Bias_occupation.csv data_raw/zhatebench/
```

3. **验证文件存在：**

```bash
ls -lh data_raw/zhatebench/
```

应该看到 6 个 CSV 文件。

## 步骤 2: 构建检测器数据集 / Step 2: Build Detector Datasets

运行数据构建脚本：

```bash
python scripts/build_detectors_datasets.py --seed 42
```

**参数说明 / Parameters:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--zhatebench-dir` | `data_raw/zhatebench` | ZHateBench 数据目录 |
| `--output-dir` | `data_processed` | 输出目录 |
| `--seed` | `42` | 随机种子（确保可复现）|
| `--train-ratio` | `0.7` | 训练集比例 |
| `--dev-ratio` | `0.15` | 验证集比例 |
| `--test-ratio` | `0.15` | 测试集比例 |
| `--hf-cache-dir` | `None` | HuggingFace 缓存目录（可选）|

**示例：自定义切分比例**

```bash
python scripts/build_detectors_datasets.py \
  --seed 42 \
  --train-ratio 0.8 \
  --dev-ratio 0.1 \
  --test-ratio 0.1
```

**脚本功能：**

1. 加载 ZHateBench 6 个 CSV 文件
2. 从 HuggingFace 加载 spam 数据集
3. 映射到统一 schema（content, toxic, source, keyword）
4. 执行数据清洗（去重、去除空值）
5. 分层切分为 train/dev/test（stratified split，确保正负样本比例一致）
6. 输出到 `data_processed/` 目录
7. 生成 `manifest.json` 元数据

## 步骤 3: 查看生成的数据 / Step 3: Verify Generated Data

**查看 manifest.json：**

```bash
cat data_processed/manifest.json
```

**查看生成的文件：**

```bash
ls -R data_processed/
```

**检查数据样本：**

```bash
head data_processed/d1_porn/train.csv
```

## 步骤 4: 训练检测器 / Step 4: Train Detectors

使用新的参数训练检测器：

### 训练 D1 (涉黄检测器) / Train D1 (Porn Detector)

```bash
python src/train.py \
  --train_csv data_processed/d1_porn/train.csv \
  --dev_csv data_processed/d1_porn/dev.csv \
  --test_csv data_processed/d1_porn/test.csv \
  --output_dir outputs/d1_porn \
  --epochs 3 \
  --batch_size 32 \
  --seed 42 \
  --tune-threshold
```

### 训练 D2 (辱骂检测器) / Train D2 (Abuse Detector)

```bash
python src/train.py \
  --train_csv data_processed/d2_abuse/train.csv \
  --dev_csv data_processed/d2_abuse/dev.csv \
  --test_csv data_processed/d2_abuse/test.csv \
  --output_dir outputs/d2_abuse \
  --epochs 3 \
  --batch_size 32 \
  --seed 42 \
  --tune-threshold
```

### 训练 D3 (歧视检测器) / Train D3 (Bias Detector)

```bash
python src/train.py \
  --train_csv data_processed/d3_bias/train.csv \
  --dev_csv data_processed/d3_bias/dev.csv \
  --test_csv data_processed/d3_bias/test.csv \
  --output_dir outputs/d3_bias \
  --epochs 3 \
  --batch_size 32 \
  --seed 42 \
  --tune-threshold
```

### 训练 D4 (广告诈骗检测器) / Train D4 (Spam Detector)

```bash
python src/train.py \
  --train_csv data_processed/d4_spam/train.csv \
  --dev_csv data_processed/d4_spam/dev.csv \
  --test_csv data_processed/d4_spam/test.csv \
  --output_dir outputs/d4_spam \
  --epochs 3 \
  --batch_size 32 \
  --seed 42 \
  --tune-threshold
```

## 训练输出 / Training Outputs

每个检测器训练完成后，会在 `outputs/{detector_name}/` 目录生成：

```
outputs/d1_porn/
├── model/                    # 模型权重
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer files...
├── metrics_dev.json          # 验证集指标
├── metrics_test.json         # 测试集指标
├── best_threshold.json       # 最佳阈值（如果使用 --tune-threshold）
├── threshold_scan.json       # 阈值扫描结果
├── test_predictions.csv      # 测试集预测结果
└── logs/                     # 训练日志
```

## 向后兼容 / Backward Compatibility

训练脚本仍然支持旧的单一 CSV 模式：

```bash
# 旧模式（仍然有效）
python src/train.py --csv_path data/ToxiCN_1.0.csv
```

## Ensemble 集成（未来）/ Ensemble (Future)

当前阶段训练 4 个独立检测器。未来可以：

1. **并行推理**: 对输入文本同时运行 4 个检测器
2. **融合策略**: 
   - 投票机制（majority voting）
   - 加权平均（weighted average）
   - 分类优先级（category priority）
3. **元分类器**: 训练一个元模型融合 4 个检测器的输出

**建议接口（示例）:**

```python
# 伪代码示例
class EnsembleDetector:
    def __init__(self, d1_model, d2_model, d3_model, d4_model):
        self.detectors = {
            'd1': d1_model,
            'd2': d2_model,
            'd3': d3_model,
            'd4': d4_model
        }
    
    def predict(self, text):
        results = {}
        for name, detector in self.detectors.items():
            results[name] = detector.predict(text)
        
        # 融合逻辑
        final_prob = self.merge_predictions(results)
        category = self.determine_category(results)
        
        return {
            'final_prob': final_prob,
            'category': category,
            'detector_results': results
        }
```

## 统一 Schema / Unified Schema

所有处理后的 CSV 文件使用统一列结构：

| 列名 | 类型 | 说明 |
|------|------|------|
| `content` | str | 文本内容 |
| `toxic` | int | 标签 (0=安全, 1=有害) |
| `source` | str | 数据源标识 (如 `ZHateBench:SexHarmSet`) |
| `keyword` | str | 关键词（可选，用于可解释性）|

## manifest.json 示例 / manifest.json Example

```json
{
  "build_date": "2026-02-16T06:30:00",
  "seed": 42,
  "zhatebench_dir": "data_raw/zhatebench",
  "output_dir": "data_processed",
  "detectors": {
    "d1_porn": {
      "train": {
        "samples": 7000,
        "toxic": 3500,
        "toxic_ratio": 0.5
      },
      "dev": {
        "samples": 1500,
        "toxic": 750,
        "toxic_ratio": 0.5
      },
      "test": {
        "samples": 1500,
        "toxic": 750,
        "toxic_ratio": 0.5
      },
      "total_samples": 10000,
      "sources": ["ZHateBench:SexHarmSet"]
    },
    "d3_bias": {
      "train": {
        "samples": 14000,
        "toxic": 7000,
        "toxic_ratio": 0.5
      },
      "sources": [
        "ZHateBench:Bias_region",
        "ZHateBench:BiasSet_genden",
        "ZHateBench:Bias_race",
        "ZHateBench:Bias_occupation"
      ]
    }
  },
  "citations": {
    "ZHateBench": {
      "doi": "10.5281/zenodo.16812052",
      "url": "https://doi.org/10.5281/zenodo.16812052",
      "warning": "⚠️ Contains sensitive content, research use only"
    },
    "spam_dataset": {
      "huggingface_id": "reatiny/chinese-spam-10000"
    }
  }
}
```

## 故障排查 / Troubleshooting

### 问题 1: ZHateBench 文件未找到

**错误:**
```
❌ ZHateBench 文件不存在: data_raw/zhatebench/SexHarmSet.csv
```

**解决:**
1. 确认文件已复制到正确目录
2. 检查文件名大小写是否匹配
3. 验证文件权限

### 问题 2: HuggingFace 数据集下载失败

**错误:**
```
ConnectionError: Couldn't reach https://huggingface.co
```

**解决:**
1. 检查网络连接
2. 使用代理（如果需要）
3. 指定缓存目录: `--hf-cache-dir /path/to/cache`

### 问题 3: CSV 列名不匹配

**错误:**
```
ValueError: CSV 文件缺少必需列: ['Type']
```

**解决:**
确认 CSV 文件包含必需列: `Keyword`, `Type`, `Sentence`

## 最佳实践 / Best Practices

1. **固定随机种子**: 始终使用相同的 `--seed` 确保可复现性
2. **保存 manifest**: 记录构建日期和参数以便追溯
3. **版本控制**: 对 `manifest.json` 进行版本控制
4. **数据备份**: 定期备份 `data_raw/` 和 `data_processed/`
5. **阈值校准**: 训练时使用 `--tune-threshold` 自动优化阈值

## 相关文档 / Related Documentation

- [README.md](README.md) - 项目总览
- [ARCHITECTURE.md](ARCHITECTURE.md) - 系统架构
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
