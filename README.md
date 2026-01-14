# 中文评论审核系统 (Chinese Comment Moderation System)

基于 Transformer 的中文有毒评论分类系统，面向 ToxiCN 数据集的 `toxic` 二分类任务，集成规则模块用于广告导流检测。

## ⚠️ 重要声明

**本项目仅供科研和学术用途，不得用于商业目的。**

- **数据集**: ToxiCN 1.0
- **许可**: CC BY-NC-ND 4.0 (非商业-禁止演绎)
- **引用**: ACL 2023
- **内容警告**: 数据集包含有毒/攻击性内容，仅用于科研目的

### ToxiCN 引用

如果您使用本系统或 ToxiCN 数据集，请引用：

```bibtex
@inproceedings{zhou-etal-2023-toxicn,
    title = "{T}oxic{CN}: A Dataset for Detecting Toxic Content in {C}hinese Conversations",
    author = "Zhou, Hao and others",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics",
    year = "2023"
}
```

## 📋 项目概述

本系统实现了端到端的中文评论审核流程：

1. **训练模块** (`src/train.py`): 基于 `hfl/chinese-roberta-wwm-ext` 微调二分类模型
2. **推理模块** (`src/predict.py`): 可复用的预测函数，支持单条/批量推理
3. **规则模块** (`src/rules.py`): 正则表达式检测广告导流（URL/微信/QQ/手机号等）
4. **FastAPI 服务** (`api/main.py`): REST API，支持单条和批量CSV预测
5. **Streamlit 界面** (`app/app.py`): Web UI 管理界面

### 系统特性

- ✅ 二分类任务（toxic 标签，sigmoid 单输出）
- ✅ 规则融合：模型预测 + 广告检测规则
- ✅ 可配置阈值（默认 0.5）
- ✅ 输出包含 `model_prob`, `rule_hits`, `rule_score`, `final_prob`, `pred`
- ✅ 支持批量预测和CSV文件处理

## 🏗️ 项目结构

```
.
├── api/
│   └── main.py              # FastAPI 服务
├── app/
│   └── app.py               # Streamlit 界面
├── src/
│   ├── train.py             # 训练脚本
│   ├── predict.py           # 推理模块
│   ├── rules.py             # 规则检测模块
│   └── data.py              # 数据加载工具
├── data/
│   └── .gitkeep             # 数据目录（CSV文件不提交）
├── outputs/
│   └── .gitkeep             # 输出目录（模型和结果不提交）
├── requirements.txt         # Python 依赖
├── .gitignore               # Git 忽略规则
└── README.md                # 本文件
```

## 🚀 快速开始

### 1. 环境配置（Conda）

**创建 Conda 环境:**

```bash
# 创建新环境（Python 3.10 推荐）
conda create -n memon python=3.10 -y

# 激活环境
conda activate memon

# 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8 示例:
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 或者 CPU 版本:
# conda install pytorch cpuonly -c pytorch -y

# 安装其他依赖
pip install -r requirements.txt
```

**快速一键安装:**

```bash
conda create -n memon python=3.10 -y && \
conda activate memon && \
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
pip install -r requirements.txt
```

### 2. 准备数据

**下载 ToxiCN 数据集并放置到 `data/` 目录:**

```bash
# 将 ToxiCN_1.0.csv 复制到 data/ 目录
cp /path/to/ToxiCN_1.0.csv data/

# 验证文件存在
ls data/ToxiCN_1.0.csv
```

**数据格式要求:**

- CSV 文件包含至少两列: `content` (文本) 和 `toxic` (标签 0/1)
- 文件编码: UTF-8

### 3. 训练模型

**基础训练（使用默认参数）:**

```bash
python src/train.py
```

**自定义参数训练:**

```bash
python src/train.py \
  --csv_path data/ToxiCN_1.0.csv \
  --model_name hfl/chinese-roberta-wwm-ext \
  --output_dir outputs \
  --epochs 3 \
  --batch_size 32 \
  --lr 2e-5 \
  --max_length 128 \
  --seed 42 \
  --threshold 0.5
```

**训练参数说明:**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--csv_path` | ToxiCN 数据集路径 | `data/ToxiCN_1.0.csv` |
| `--model_name` | 预训练模型名称 | `hfl/chinese-roberta-wwm-ext` |
| `--output_dir` | 输出目录 | `outputs` |
| `--max_length` | 最大序列长度 | 128 |
| `--epochs` | 训练轮数 | 3 |
| `--batch_size` | 训练批次大小 | 32 |
| `--lr` | 学习率 | 2e-5 |
| `--seed` | 随机种子 | 42 |
| `--threshold` | 预测阈值 | 0.5 |

**训练输出:**

训练完成后，会在 `outputs/` 目录下生成：

```
outputs/
├── model/                    # 保存的模型和tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
├── metrics_dev.json          # 验证集指标
├── metrics_test.json         # 测试集指标
└── test_predictions.csv      # 测试集预测结果
```

### 4. 启动 FastAPI 服务

**启动 API 服务器:**

```bash
# 方式1: 使用 uvicorn（推荐用于开发）
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 方式2: 直接运行
python api/main.py

# 方式3: 生产环境（多worker）
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**访问 API 文档:**

打开浏览器访问: http://localhost:8000/docs

### 5. 启动 Streamlit 界面

**启动 Web UI:**

```bash
streamlit run app/app.py
```

默认会在浏览器自动打开: http://localhost:8501

## 📡 API 使用示例

### 单条预测

**请求示例 (curl):**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这个产品很好用，推荐大家购买",
    "threshold": 0.5
  }'
```

**响应示例:**

```json
{
  "text": "这个产品很好用，推荐大家购买",
  "model_prob": 0.023,
  "rule_hits": [],
  "rule_score": 0.0,
  "final_prob": 0.023,
  "pred": 0,
  "threshold": 0.5
}
```

**广告检测示例:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "加微信VX123456了解详情，低价批发",
    "threshold": 0.5
  }'
```

**响应:**

```json
{
  "text": "加微信VX123456了解详情，低价批发",
  "model_prob": 0.156,
  "rule_hits": ["WeChat", "Price"],
  "rule_score": 1.0,
  "final_prob": 1.0,
  "pred": 1,
  "threshold": 0.5
}
```

### 批量预测

**上传 CSV 文件:**

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -F "file=@test_comments.csv" \
  -F "threshold=0.5" \
  -F "text_column=content" \
  -o predictions.csv
```

**Python 示例:**

```python
import requests

# 单条预测
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "这个产品很好用",
        "threshold": 0.5
    }
)
print(response.json())

# 批量预测
with open('test_comments.csv', 'rb') as f:
    files = {'file': f}
    data = {'threshold': 0.5, 'text_column': 'content'}
    response = requests.post(
        "http://localhost:8000/batch_predict",
        files=files,
        data=data
    )
    
    # 保存结果
    with open('predictions.csv', 'wb') as out:
        out.write(response.content)
```

## 🎨 Streamlit 界面功能

### 单条预测

1. 输入评论文本
2. 调整判定阈值（侧边栏）
3. 点击"预测"按钮
4. 查看结果：
   - 模型概率
   - 规则分数
   - 最终概率
   - 判定结果（有毒/正常）
   - 规则命中列表

### 批量预测

1. 上传 CSV 文件
2. 选择文本列名
3. 点击"批量预测"
4. 查看统计信息和预览
5. 下载完整预测结果

## 🧪 规则检测模块

规则模块 (`src/rules.py`) 检测以下广告导流模式：

| 规则类型 | 检测内容 | 示例 |
|---------|---------|------|
| URL | 网址链接 | `http://example.com`, `www.site.com` |
| WeChat | 微信号/VX | `微信`, `VX123456`, `威信` |
| QQ | QQ号/群 | `QQ:123456789`, `扣扣群` |
| Phone | 手机号 | `13812345678` |
| Price | 价格/优惠 | `低价`, `优惠`, `返现`, `仅需99元` |
| Group | 加群引导 | `加群`, `进群`, `群号` |
| Scam | 刷单/兼职 | `刷单`, `兼职`, `日赚千元` |
| Contact | 联系引导 | `私信我`, `详情咨询` |

**融合策略:**

- 默认: `final_prob = max(model_prob, rule_score)`
- 可配置规则覆盖 (`RULE_OVERRIDE=True`): 规则命中直接判定为有毒

**规则分数计算:**

- 命中 0 个规则: `score = 0.0`
- 命中 1 个规则: `score = 0.6`
- 命中 2+ 个规则: `score = 1.0`

## 📊 输出格式

### 测试集预测文件 (`outputs/test_predictions.csv`)

| 列名 | 说明 |
|------|------|
| `content` | 原始文本 |
| `label` | 真实标签 (0/1) |
| `model_prob` | 模型预测概率 |
| `rule_score` | 规则得分 |
| `final_prob` | 融合后最终概率 |
| `pred` | 预测标签 (0/1) |
| `rule_hits` | 命中的规则列表 (逗号分隔) |

## 🛠️ 开发和调试

### 测试各模块

```bash
# 测试规则模块
python src/rules.py

# 测试数据加载
python src/data.py

# 测试预测模块（需要先训练模型）
python src/predict.py
```

### 常见问题

**Q: 模型下载慢或失败？**

A: 设置 Hugging Face 镜像源：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或在代码中设置代理。

**Q: CUDA 内存不足？**

A: 减小 `batch_size` 或 `max_length`：

```bash
python src/train.py --batch_size 16 --max_length 64
```

**Q: 找不到模型文件？**

A: 确保先运行训练脚本生成模型：

```bash
python src/train.py
ls outputs/model/  # 检查模型文件
```

**Q: TrainingArguments 参数错误 (evaluation_strategy vs eval_strategy)？**

A: 本系统已内置兼容逻辑，支持 `transformers` 新旧版本：
- **旧版本** (< 4.19.0): 使用 `evaluation_strategy` 参数
- **新版本** (>= 4.19.0): 使用 `eval_strategy` 参数

如果仍遇到问题，建议升级 transformers：

```bash
pip install --upgrade transformers
```

系统会自动检测当前版本并使用正确的参数名，无需手动配置。

## 📈 性能指标

训练完成后，查看评估指标：

```bash
# 验证集指标
cat outputs/metrics_dev.json

# 测试集指标
cat outputs/metrics_test.json
```

指标包括：
- Accuracy (准确率)
- Precision (精确率)
- Recall (召回率)
- F1 Score (F1分数)
- AUC (ROC曲线下面积)

## 🔒 安全和隐私

- ❌ 数据文件 (`data/*.csv`) **不会**提交到 Git 仓库
- ❌ 模型权重 (`outputs/`) **不会**提交到 Git 仓库
- ✅ 所有敏感文件已在 `.gitignore` 中配置
- ⚠️ 请勿将 API 暴露到公网，仅用于本地/内网测试

## 📝 许可和引用

### 数据集许可

ToxiCN 数据集采用 **CC BY-NC-ND 4.0** 许可：
- ✅ 允许科研和学术使用
- ❌ 禁止商业使用
- ❌ 禁止演绎修改（仅可复制和分发）

### 系统许可

本系统代码开源，供学习和科研使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请通过 GitHub Issues 联系。

---

**最后更新**: 2026-01-14