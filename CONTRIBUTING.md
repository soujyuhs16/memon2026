# 开发指南 (Development Guide)

## 项目开发流程

### 1. 设置开发环境

```bash
# 克隆仓库
git clone https://github.com/soujyuhs16/memon2026.git
cd memon2026

# 创建 conda 环境
conda create -n memon python=3.10 -y
conda activate memon

# 安装 PyTorch
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装依赖
pip install -r requirements.txt
```

### 2. 代码风格

本项目遵循 PEP 8 Python 代码风格指南。

**推荐工具:**
- `black`: 代码格式化
- `flake8`: 代码检查
- `pylint`: 静态分析

```bash
# 安装开发工具
pip install black flake8 pylint

# 格式化代码
black src/ api/ app/

# 检查代码
flake8 src/ api/ app/ --max-line-length=100
```

### 3. 模块说明

#### src/rules.py - 规则检测模块

**功能**: 基于正则表达式检测广告导流模式

**扩展规则**: 在 `check_rules()` 函数中添加新的正则模式

```python
# 示例：添加新规则
if re.search(r'(新关键词|另一个模式)', text):
    hits.append('NewRule')
```

#### src/data.py - 数据加载模块

**功能**: 加载和预处理 ToxiCN 数据集

**扩展**: 支持其他数据格式或数据增强

#### src/predict.py - 推理模块

**功能**: 提供可复用的模型推理接口

**ToxicClassifier 类**:
- `predict_single()`: 单条预测
- `predict_batch()`: 批量预测

#### src/train.py - 训练脚本

**功能**: 使用 Hugging Face Transformers 训练模型

**自定义训练**: 修改 `TrainingArguments` 配置

#### api/main.py - FastAPI 服务

**功能**: REST API 服务

**添加新端点**: 使用 `@app.post()` 或 `@app.get()` 装饰器

```python
@app.get("/new_endpoint")
def new_endpoint():
    return {"message": "Hello"}
```

#### app/app.py - Streamlit 界面

**功能**: Web UI 管理界面

**自定义界面**: 使用 Streamlit 组件扩展功能

### 4. 测试

#### 单元测试

```bash
# 测试规则模块
python src/rules.py

# 测试数据模块
python src/data.py
```

#### 集成测试

```bash
# 运行集成测试脚本
./test_integration.sh
```

#### 端到端测试

```bash
# 1. 准备小数据集
# 2. 训练模型
python src/train.py --epochs 1 --batch_size 8

# 3. 启动 API
uvicorn api.main:app --reload &

# 4. 测试 API
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "测试文本"}'

# 5. 启动 UI
streamlit run app/app.py
```

### 5. 提交代码

```bash
# 添加修改
git add .

# 提交（使用有意义的提交信息）
git commit -m "feat: 添加新功能"

# 推送
git push origin branch-name
```

**提交信息格式建议:**
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具相关

### 6. 性能优化建议

#### 模型推理优化

1. **批量处理**: 使用 `predict_batch()` 而不是循环调用 `predict_single()`
2. **GPU 加速**: 确保模型在 GPU 上运行
3. **模型量化**: 使用 ONNX 或 TorchScript 优化
4. **缓存**: 对频繁查询的结果进行缓存

#### API 优化

1. **异步处理**: 使用 FastAPI 的异步特性
2. **连接池**: 复用模型实例
3. **限流**: 添加请求限流保护

### 7. 常见问题

**Q: 如何更换基线模型？**

A: 修改训练脚本的 `--model_name` 参数：

```bash
python src/train.py --model_name bert-base-chinese
```

**Q: 如何调整规则权重？**

A: 修改 `src/rules.py` 中的评分逻辑或 `merge_predictions()` 函数。

**Q: 如何支持多标签分类？**

A: 需要修改：
1. `src/train.py`: 改为多标签损失函数
2. `src/predict.py`: 改为多输出 sigmoid
3. API 和 UI: 支持多标签返回

### 8. 文档

- 代码注释：使用中英文双语注释
- Docstring：遵循 Google 风格
- README：保持更新

### 9. 许可和合规

- **数据集**: 仅用于科研，遵守 CC BY-NC-ND 4.0
- **代码**: 开源，供学习研究使用
- **不得**: 商业使用、传播敏感内容

## 联系方式

如有问题，请通过 GitHub Issues 联系。
