# 数据清洗与阈值校准指南

本文档介绍数据清洗和阈值校准功能的使用方法。

## 数据清洗

### 快速开始

```bash
# 基础清洗：剔除CHSD中toxic=0但包含有害内容的样本
python scripts/build_dataset.py --mode drop

# 加入clean-negative样本
python scripts/build_dataset.py --mode drop --clean-neg-ratio 0.05
```

### 参数说明

- `--input`: 输入CSV路径（默认：data/mixture_toxicn_chsd.csv）
- `--output`: 输出CSV路径（默认：data/mixture_cleaned.csv）
- `--mode`: drop=剔除 或 relabel=翻标
- `--clean-neg-ratio`: 加入clean-negative比例（0.0-1.0）
- `--relabel-high-conf`: 仅翻标高置信样本

## 阈值校准

### 快速开始

```bash
# 训练时启用阈值校准
python src/train.py --csv_path data/mixture_cleaned.csv --tune-threshold

# 指定策略
python src/train.py --csv_path data/mixture_cleaned.csv --tune-threshold --threshold-strategy max_recall_min_precision --min-precision 0.8
```

### 校准策略

1. `max_f1`（默认）：最大化F1分数
2. `max_recall_min_precision`：在满足最小precision下最大化recall

### 输出文件

- `outputs/best_threshold.json`: 最佳阈值及指标
- `outputs/threshold_scan.json`: 全部阈值扫描结果

**注意**: 旧版本使用 `threshold.json`，新版本改为 `best_threshold.json`。推理模块会自动兼容两种文件名。

## 完整工作流

```bash
# 步骤1: 数据清洗
python scripts/build_dataset.py --mode drop --clean-neg-ratio 0.05

# 步骤2: 训练+阈值校准
python src/train.py --csv_path data/mixture_cleaned.csv --tune-threshold

# 步骤3: 推理（自动使用校准阈值）
python -c "from src.predict import load_predictor; p=load_predictor('outputs/model'); print(p.predict_one('测试'))"
```

## 推理使用

推理模块会自动加载 `outputs/best_threshold.json` (或 `threshold.json`，向后兼容) 中的校准阈值。

### Python接口

```python
from src.predict import load_predictor

predictor = load_predictor('outputs/model')
result = predictor.predict_one('测试文本')  # 使用校准阈值
result = predictor.predict_one('测试文本', threshold=0.6)  # 手动覆盖
```

### API和Web UI

- FastAPI和Streamlit会自动使用校准阈值
- 用户可手动调整阈值参数

---

更新时间: 2026-02-16
