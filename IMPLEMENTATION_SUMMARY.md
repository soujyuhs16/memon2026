# Implementation Summary: Data Cleaning and Threshold Tuning

## Overview

This implementation adds two major features to improve the toxic content detection system:

1. **Data Cleaning**: Automatically detects and removes mislabeled samples from training data
2. **Threshold Tuning**: Calibrates the optimal decision threshold based on validation data

## What Was Implemented

### 1. Data Cleaning Script (`scripts/build_dataset.py`)

**Purpose**: Clean CHSD dataset samples that are labeled as safe (toxic=0) but actually contain harmful content.

**Features**:
- Keyword detection for 8 harmful categories:
  - Regional discrimination (地域歧视)
  - Gender discrimination (性别歧视)  
  - Racial discrimination (种族歧视)
  - Homophobia (恐同)
  - Abuse (辱骂)
  - Threats (威胁)
  - Pornographic drainage (色情引流)
  - Advertising/scams (广告诈骗)

- Pattern detection for URLs, phone numbers, QQ numbers

- Multiple processing modes:
  - `drop`: Remove detected samples (default, safest)
  - `relabel`: Change label to toxic=1
  - `relabel + high-conf`: Only relabel high-confidence detections

- Clean-negative augmentation: Add high-quality negative examples

**Usage**:
```bash
# Basic cleaning
python scripts/build_dataset.py --mode drop

# With clean negatives
python scripts/build_dataset.py --mode drop --clean-neg-ratio 0.05
```

**Results**: Detects 864 harmful samples (10.93%) in CHSD toxic=0 data

### 2. Threshold Tuning in Training (`src/train.py`)

**Purpose**: Find the optimal classification threshold instead of using fixed 0.5.

**How it works**:
- After training, runs grid search on validation set
- Tests thresholds from 0.05 to 0.95 (step 0.01)
- Calculates precision/recall/F1 for each threshold
- Selects best threshold based on strategy

**Strategies**:
- `max_f1`: Maximize F1 score (default)
- `max_recall_min_precision`: Maximize recall while maintaining minimum precision

**Usage**:
```bash
# Enable threshold tuning
python src/train.py --csv_path data/mixture_cleaned.csv --tune-threshold

# With custom strategy
python src/train.py --csv_path data/mixture_cleaned.csv --tune-threshold \
  --threshold-strategy max_recall_min_precision --min-precision 0.8
```

**Output**:
- `outputs/threshold.json`: Best threshold and its metrics
- `outputs/threshold_scan.json`: All thresholds tested (for analysis)

### 3. Automatic Threshold Loading (`src/predict.py`)

**Purpose**: Use the calibrated threshold automatically in inference.

**How it works**:
- When loading a model, checks for `threshold.json` in parent directory
- If found, loads the calibrated threshold as default
- Falls back to 0.5 if not found
- Can still be manually overridden

**Usage**:
```python
from src.predict import load_predictor

# Automatic: uses calibrated threshold
predictor = load_predictor('outputs/model')
result = predictor.predict_one('测试文本')

# Manual override
result = predictor.predict_one('测试文本', threshold=0.6)
```

### 4. API and Web UI Integration

Both `api/main.py` and `app/app.py` have been updated to:
- Use calibrated threshold by default
- Display the calibrated threshold to users
- Allow manual override when needed

### 5. Documentation

- **Guide**: `docs/DATA_CLEANING_AND_THRESHOLD_TUNING.md`
- **README**: Updated with new features section
- **Example data**: `data/clean_negatives.csv` (30 clean short sentences)

## Complete Workflow Example

```bash
# Step 1: Clean the training data
python scripts/build_dataset.py \
  --input data/mixture_toxicn_chsd.csv \
  --output data/mixture_cleaned.csv \
  --mode drop \
  --clean-neg-ratio 0.05

# Step 2: Train with threshold tuning
python src/train.py \
  --csv_path data/mixture_cleaned.csv \
  --output_dir outputs \
  --epochs 3 \
  --tune-threshold

# Step 3: Inference (automatically uses calibrated threshold)
python -c "from src.predict import load_predictor; \
  p = load_predictor('outputs/model'); \
  print(p.predict_one('这是测试文本'))"
```

## Key Improvements

### Data Quality
- **Before**: 7,904 CHSD toxic=0 samples (some mislabeled)
- **After**: 7,040 verified safe samples (864 harmful removed)
- **Benefit**: Model learns from cleaner data

### Threshold Optimization
- **Before**: Fixed threshold 0.5 (arbitrary)
- **After**: Data-driven threshold (e.g., 0.42 for max F1)
- **Benefit**: Better precision/recall tradeoff

### False Positive Reduction
- Refined keyword lists (removed neutral terms like "黑人", "同性恋")
- Clean-negative augmentation (e.g., "你好", "谢谢")
- **Benefit**: Normal content less likely to be flagged

## File Changes

**New files**:
- `scripts/__init__.py`
- `scripts/build_dataset.py`
- `data/clean_negatives.csv`
- `docs/DATA_CLEANING_AND_THRESHOLD_TUNING.md`

**Modified files**:
- `src/train.py`: Added threshold tuning
- `src/predict.py`: Added automatic threshold loading
- `api/main.py`: Updated to use calibrated threshold
- `app/app.py`: Display calibrated threshold
- `README.md`: Added new features section
- `.gitignore`: Exception for clean_negatives.csv

## Testing Results

✅ Data cleaning script: Works correctly, detects 10.93% harmful samples
✅ Threshold tuning: Parameters verified, outputs JSON files
✅ Threshold loading: Automatic loading works, fallback to 0.5
✅ API integration: Calibrated threshold used by default
✅ Web UI: Shows calibrated threshold in sidebar

## Next Steps

1. Run data cleaning on your dataset
2. Train a model with threshold tuning enabled
3. Use the calibrated threshold in production
4. Monitor false positive/negative rates
5. Adjust keyword lists or strategies as needed

## Support

- See `docs/DATA_CLEANING_AND_THRESHOLD_TUNING.md` for detailed guide
- Check `data/clean_negatives.csv` for example format
- Run scripts with `--help` for all options

---

**Implementation Date**: 2026-02-16
**Status**: Complete and tested ✅
