# 数据管线实现总结 / Data Pipeline Implementation Summary

## 概述 / Overview

本 PR 实现了完整的多检测器数据管线，支持从 ZHateBench 和 HuggingFace 数据集构建四个检测器（D1-D4）的训练数据，并更新训练脚本支持显式的 train/dev/test 切分。

This PR implements a complete multi-detector data pipeline supporting the construction of training data for four detectors (D1-D4) from ZHateBench and HuggingFace datasets, and updates the training script to support explicit train/dev/test splits.

## 主要变更 / Key Changes

### 1. 数据加载工具 / Data Loading Utilities

**新增文件:**
- `src/data/zhatebench.py`: ZHateBench 数据集加载工具
- `src/data/build_splits.py`: 分层数据切分工具

**功能:**
- 加载 6 个 ZHateBench CSV 文件（涉黄、辱骂、4种歧视类型）
- 统一数据 schema: `content`, `toxic`, `source`, `keyword`
- 数据清洗：去重、去除空值
- 分层切分：确保 train/dev/test 中正负样本比例一致

### 2. 数据构建脚本 / Dataset Building Script

**新增文件:**
- `scripts/build_detectors_datasets.py`: 主数据构建脚本

**功能:**
- 加载 ZHateBench 6 个类别
- 从 HuggingFace 加载 spam 数据集（`reatiny/chinese-spam-10000`）
- 网络失败时使用模拟数据作为备用
- 按检测器分组：
  - D1: 涉黄 (SexHarmSet)
  - D2: 辱骂 (AbuseSet)
  - D3: 歧视 (合并 4 个 bias CSV)
  - D4: 广告诈骗 (spam 数据集)
- 生成 train/dev/test 切分（固定种子，可复现）
- 输出 `manifest.json` 元数据文件

**命令示例:**
```bash
python scripts/build_detectors_datasets.py --seed 42
```

### 3. 训练脚本更新 / Training Script Updates

**更新文件:**
- `src/train.py`: 支持显式 train/dev/test CSV 参数

**新功能:**
- **模式 1 (向后兼容)**: `--csv_path` 单一 CSV，自动切分
- **模式 2 (推荐)**: `--train_csv`, `--dev_csv`, `--test_csv` 显式指定
- 阈值文件名改为 `best_threshold.json`（更具描述性）

**命令示例:**
```bash
# 新模式：显式 train/dev/test
python src/train.py \
  --train_csv data_processed/d1_porn/train.csv \
  --dev_csv data_processed/d1_porn/dev.csv \
  --test_csv data_processed/d1_porn/test.csv \
  --output_dir outputs/d1_porn \
  --tune-threshold

# 旧模式（仍支持）
python src/train.py --csv_path data/ToxiCN_1.0.csv
```

### 4. 推理模块更新 / Inference Module Updates

**更新文件:**
- `src/predict.py`: 支持新旧阈值文件名

**功能:**
- 优先加载 `best_threshold.json`
- 回退到 `threshold.json`（向后兼容）

### 5. 文档 / Documentation

**新增文件:**
- `DATA_PIPELINE.md`: 完整的数据管线使用指南
  - 数据来源说明（ZHateBench, spam dataset）
  - 引用信息（DOI, HuggingFace ID）
  - 使用步骤（准备数据、构建、训练）
  - 故障排查
  - Ensemble 集成建议

**更新文件:**
- `README.md`: 添加多检测器架构概述
- `docs/DATA_CLEANING_AND_THRESHOLD_TUNING.md`: 更新阈值文件名说明

### 6. 测试 / Tests

**新增文件:**
- `tests/test_data_pipeline.py`: 数据管线功能测试
  - 参数解析测试
  - 数据加载测试

**测试覆盖:**
- ✅ ZHateBench 加载
- ✅ HuggingFace 加载（含网络失败备用方案）
- ✅ 数据切分（分层抽样）
- ✅ 训练参数解析（两种模式）
- ✅ 数据列结构验证

### 7. 目录结构 / Directory Structure

**新增目录:**
```
data_raw/
└── zhatebench/          # ZHateBench 原始数据
    ├── SexHarmSet.csv
    ├── AbuseSet.csv
    ├── Bias_region.csv
    ├── BiasSet_genden.csv
    ├── Bias_race.csv
    └── Bias_occupation.csv

data_processed/          # 处理后的数据
├── d1_porn/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
├── d2_abuse/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
├── d3_bias/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
├── d4_spam/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
└── manifest.json
```

## 数据源引用 / Data Source Citations

### ZHateBench
```
DOI: 10.5281/zenodo.16812052
URL: https://doi.org/10.5281/zenodo.16812052
⚠️ 包含敏感内容，仅供研究使用
Contains sensitive content, research use only
```

### Spam Dataset
```
HuggingFace ID: reatiny/chinese-spam-10000
URL: https://huggingface.co/datasets/reatiny/chinese-spam-10000
```

## 技术特性 / Technical Features

1. **可复现性 / Reproducibility**
   - 固定随机种子（默认 42）
   - 分层抽样确保数据分布一致
   - 元数据记录（manifest.json）

2. **数据质量 / Data Quality**
   - 去重（同一数据集内和跨数据集）
   - 空值移除
   - 统一 schema

3. **向后兼容 / Backward Compatibility**
   - 保留旧的 `--csv_path` 参数
   - 支持旧的 `threshold.json` 文件名
   - 不影响现有训练流程

4. **错误处理 / Error Handling**
   - 网络失败时使用模拟数据
   - 清晰的错误提示
   - 文件缺失检查

5. **文档完善 / Documentation**
   - 中英双语
   - 详细的使用示例
   - 故障排查指南

## 验收标准 / Acceptance Criteria

- ✅ 运行 `python scripts/build_detectors_datasets.py --seed 42` 能生成所有数据文件
- ✅ 生成的 CSV 包含正确的列：`content`, `toxic`, `source`, `keyword`
- ✅ manifest.json 包含完整的元数据和引用信息
- ✅ 训练脚本支持 `--train_csv`, `--dev_csv`, `--test_csv` 参数
- ✅ 训练脚本保持向后兼容（`--csv_path` 仍可用）
- ✅ 阈值保存为 `best_threshold.json`
- ✅ 推理模块兼容新旧阈值文件名
- ✅ 文档完整，包含引用信息和使用警告
- ✅ 通过所有测试
- ✅ 通过代码审查
- ✅ 通过安全扫描（CodeQL: 0 alerts）

## 使用流程 / Usage Workflow

### 快速开始 / Quick Start

```bash
# 1. 准备 ZHateBench 数据
mkdir -p data_raw/zhatebench
cp /path/to/ZHateBench/*.csv data_raw/zhatebench/

# 2. 构建检测器数据集
python scripts/build_detectors_datasets.py --seed 42

# 3. 训练 D1 检测器
python src/train.py \
  --train_csv data_processed/d1_porn/train.csv \
  --dev_csv data_processed/d1_porn/dev.csv \
  --test_csv data_processed/d1_porn/test.csv \
  --output_dir outputs/d1_porn \
  --epochs 3 \
  --tune-threshold

# 4. 训练其他检测器 (D2, D3, D4)...
```

## 未来工作 / Future Work

1. **Ensemble 集成**
   - 实现多检测器并行推理
   - 设计融合策略（投票、加权平均）
   - 元分类器训练

2. **数据增强**
   - 添加更多数据源
   - 数据增强技术（回译、同义替换）

3. **性能优化**
   - 模型量化
   - 多 GPU 训练
   - 批量推理优化

4. **监控与评估**
   - 在线学习
   - A/B 测试框架
   - 性能监控仪表板

## 安全与隐私 / Security and Privacy

- ✅ 通过 CodeQL 安全扫描（0 alerts）
- ✅ 数据仅在学校内保存，不公开
- ✅ 明确标注数据来源和使用限制
- ✅ 无硬编码敏感信息
- ✅ 适当的文件权限和访问控制

## 致谢 / Acknowledgments

- **ZHateBench**: Chinese hate speech benchmark dataset
- **ToxiCN**: Chinese toxic content detection dataset
- **HuggingFace**: For hosting and providing easy access to datasets
- **Transformers Library**: For state-of-the-art NLP models

---

**Note**: This implementation follows best practices for research software development, including reproducibility, documentation, testing, and security considerations.
