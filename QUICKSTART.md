# 快速开始指南 (Quick Start Guide)

本指南帮助你在 5 分钟内启动中文评论审核系统。

## 🚀 5 分钟快速启动

### 步骤 1: 克隆仓库

```bash
git clone https://github.com/soujyuhs16/memon2026.git
cd memon2026
```

### 步骤 2: 创建环境并安装依赖

**使用 Conda (推荐):**

```bash
# 创建环境
conda create -n memon python=3.10 -y

# 激活环境
conda activate memon

# 安装 PyTorch (GPU版本)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 或者 CPU 版本
# conda install pytorch cpuonly -c pytorch -y

# 安装其他依赖
pip install -r requirements.txt
```

**使用 pip:**

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install torch  # 或访问 pytorch.org 选择合适版本
pip install -r requirements.txt
```

### 步骤 3: 准备数据

```bash
# 将 ToxiCN_1.0.csv 复制到 data/ 目录
cp /path/to/ToxiCN_1.0.csv data/

# 验证文件
head -5 data/ToxiCN_1.0.csv
```

**数据格式示例:**
```csv
content,toxic
这个产品很好用,0
你个傻X,1
服务态度很好,0
```

### 步骤 4: 训练模型

```bash
# 快速训练（小数据集或测试）
python src/train.py --epochs 1 --batch_size 16

# 完整训练（默认参数）
python src/train.py

# 自定义训练
python src/train.py \
  --epochs 3 \
  --batch_size 32 \
  --lr 2e-5 \
  --max_length 128
```

**训练时间估算:**
- CPU: 30-60 分钟 (取决于数据量)
- GPU (Tesla T4): 5-10 分钟

**训练完成后检查输出:**
```bash
ls outputs/
# 应该看到: model/, metrics_dev.json, metrics_test.json, test_predictions.csv
```

### 步骤 5: 启动服务

**方式 A: 启动 FastAPI 服务**

```bash
# 启动 API (开发模式)
uvicorn api.main:app --reload

# 或生产模式
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

访问 API 文档: http://localhost:8000/docs

**测试 API:**
```bash
# 健康检查
curl http://localhost:8000/health

# 单条预测
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "这个产品很好用"}'
```

**方式 B: 启动 Streamlit UI**

```bash
# 启动 Web UI
streamlit run app/app.py
```

访问界面: http://localhost:8501

### 步骤 6: 开始使用

**单条预测 (Python):**

```python
from src.predict import load_model

# 加载模型
classifier = load_model('outputs/model')

# 预测
result = classifier.predict_single(
    "这个产品很好用，推荐大家购买",
    threshold=0.5
)

print(f"文本: {result['text']}")
print(f"预测: {'有毒' if result['pred'] == 1 else '正常'}")
print(f"概率: {result['final_prob']:.3f}")
print(f"规则: {result['rule_hits']}")
```

**批量预测 (CSV):**

```python
import pandas as pd
from src.predict import load_model

# 加载模型
classifier = load_model('outputs/model')

# 读取CSV
df = pd.read_csv('test_comments.csv')

# 批量预测
texts = df['content'].tolist()
results = classifier.predict_batch(texts, threshold=0.5)

# 保存结果
result_df = pd.DataFrame(results)
result_df.to_csv('predictions.csv', index=False)
```

## 🧪 测试系统

运行集成测试:

```bash
./test_integration.sh
```

测试规则模块:

```bash
python src/rules.py
```

## 📖 更多信息

- **完整文档**: 查看 [README.md](README.md)
- **API 示例**: 查看 [examples.py](examples.py)
- **开发指南**: 查看 [CONTRIBUTING.md](CONTRIBUTING.md)
- **架构文档**: 查看 [ARCHITECTURE.md](ARCHITECTURE.md)

## 🐛 常见问题

### Q1: 模型下载慢或失败

**解决方案**: 使用国内镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
python src/train.py
```

### Q2: CUDA out of memory

**解决方案**: 减小批次大小

```bash
python src/train.py --batch_size 8 --max_length 64
```

### Q3: 找不到模型文件

**解决方案**: 确保先运行训练

```bash
# 检查模型是否存在
ls outputs/model/

# 如果不存在，重新训练
python src/train.py
```

### Q4: API 无法启动

**解决方案**: 检查依赖和模型

```bash
# 检查依赖
pip list | grep fastapi

# 检查模型
ls outputs/model/

# 重新安装依赖
pip install -r requirements.txt
```

### Q5: Streamlit 启动慢

**解决方案**: 正常现象，首次加载模型需要时间

```bash
# 查看日志
streamlit run app/app.py --server.runOnSave false
```

## 🎯 下一步

1. **调整阈值**: 根据业务需求调整判定阈值
2. **优化规则**: 在 `src/rules.py` 中添加自定义规则
3. **性能优化**: 使用 GPU 加速推理
4. **部署生产**: 使用 Docker 容器化部署

## 📞 获取帮助

- **GitHub Issues**: 提交问题和建议
- **文档**: 查看完整文档
- **社区**: 加入讨论

---

**祝使用愉快! 🎉**
