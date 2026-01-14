# 系统架构文档 (System Architecture)

## 总体架构 (Overall Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                     中文评论审核系统 MVP                          │
│              Chinese Comment Moderation System                  │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │   ToxiCN 数据集   │
                    │  (CSV 文件)      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   训练脚本        │
                    │   (train.py)     │
                    │                  │
                    │ - 数据划分        │
                    │ - 微调 RoBERTa   │
                    │ - 评估保存        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   训练好的模型    │
                    │ (outputs/model/) │
                    └────────┬─────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐
    │   FastAPI 服务    │      │  Streamlit UI    │
    │   (api/main.py)  │      │  (app/app.py)    │
    │                  │      │                  │
    │ - /predict       │      │ - 单条预测        │
    │ - /batch_predict │      │ - 批量CSV预测     │
    └────────┬─────────┘      └────────┬─────────┘
             │                         │
             └────────────┬────────────┘
                          │
                          ▼
              ┌──────────────────────┐
              │   推理模块            │
              │   (predict.py)       │
              │                      │
              │ - ToxicClassifier    │
              │ - 模型预测           │
              │ - 规则融合           │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   规则模块            │
              │   (rules.py)         │
              │                      │
              │ - 广告检测           │
              │ - 8种规则模式        │
              │ - 得分计算           │
              └──────────────────────┘
```

## 数据流 (Data Flow)

### 训练阶段 (Training Phase)

```
ToxiCN.csv
    │
    ├─► 加载数据 (data.py)
    │
    ├─► 划分数据集 (train/dev/test)
    │
    ├─► Tokenization
    │
    ├─► Trainer 微调
    │
    └─► 输出
        ├─ outputs/model/ (模型权重)
        ├─ outputs/metrics_dev.json
        ├─ outputs/metrics_test.json
        └─ outputs/test_predictions.csv
```

### 推理阶段 (Inference Phase)

```
用户输入文本
    │
    ├─► 模型推理 (ToxicClassifier)
    │   └─► model_prob (0~1)
    │
    ├─► 规则检测 (check_rules)
    │   ├─► rule_hits: ['WeChat', 'Price', ...]
    │   └─► rule_score (0~1)
    │
    └─► 融合预测 (merge_predictions)
        └─► final_prob = max(model_prob, rule_score)
            └─► pred = 1 if final_prob >= threshold else 0
```

## 模块详解 (Module Details)

### 1. 训练模块 (src/train.py)

**输入:**
- CSV 文件 (content + toxic 列)

**处理:**
1. 数据划分 (分层抽样)
2. 加载预训练模型 (`hfl/chinese-roberta-wwm-ext`)
3. Tokenization
4. Trainer 训练
5. 评估验证集和测试集
6. 生成预测文件

**输出:**
- 模型文件 (config.json, pytorch_model.bin, tokenizer)
- 指标文件 (metrics_dev.json, metrics_test.json)
- 预测文件 (test_predictions.csv)

**关键参数:**
```python
--model_name: 预训练模型
--epochs: 训练轮数
--batch_size: 批次大小
--lr: 学习率
--max_length: 最大序列长度
--threshold: 判定阈值
```

### 2. 推理模块 (src/predict.py)

**核心类: ToxicClassifier**

```python
classifier = ToxicClassifier(model_path)

# 单条预测
result = classifier.predict_single(text, threshold=0.5)
# 返回: {text, model_prob, rule_hits, rule_score, final_prob, pred}

# 批量预测
results = classifier.predict_batch(texts, threshold=0.5)
```

**特性:**
- 自动检测 GPU/CPU
- 支持批量处理
- 集成规则融合
- 可配置阈值

### 3. 规则模块 (src/rules.py)

**8 种检测规则:**

| 规则 | 模式 | 示例 |
|-----|------|------|
| URL | `https?://`, `www.` | http://example.com |
| WeChat | 微信, VX, wx | 加微信VX123456 |
| QQ | QQ号, qq群 | QQ:123456789 |
| Phone | 手机号 | 13812345678 |
| Price | 价格, 优惠, 返现 | 低价批发仅需99元 |
| Group | 加群, 入群 | 加群了解详情 |
| Scam | 刷单, 兼职 | 日赚千元 |
| Contact | 私信, 联系 | 私信我了解 |

**融合策略:**
```python
# 默认: 取最大值
final_prob = max(model_prob, rule_score)

# 可选: 规则覆盖
if RULE_OVERRIDE and rule_score > 0:
    final_prob = 1.0
```

### 4. FastAPI 服务 (api/main.py)

**端点 (Endpoints):**

```
GET  /              # 根路径，API 信息
GET  /health        # 健康检查
POST /predict       # 单条预测
POST /batch_predict # 批量CSV预测
```

**示例请求:**

```bash
# 单条预测
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "测试文本", "threshold": 0.5}'

# 批量预测
curl -X POST http://localhost:8000/batch_predict \
  -F "file=@test.csv" \
  -F "threshold=0.5" \
  -o predictions.csv
```

**特性:**
- 自动 API 文档 (/docs)
- 错误处理
- 文件上传/下载
- 健康检查

### 5. Streamlit UI (app/app.py)

**界面组件:**

1. **配置侧边栏**
   - 阈值滑块
   - 模型信息

2. **单条预测 Tab**
   - 文本输入框
   - 预测按钮
   - 结果展示 (概率、判定、规则)

3. **批量预测 Tab**
   - CSV 文件上传
   - 列名选择
   - 结果预览
   - 下载按钮

**特性:**
- 响应式布局
- 实时预测
- 批量处理
- 结果可视化

## 文件结构 (File Structure)

```
memon2026/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI 服务
├── app/
│   ├── __init__.py
│   └── app.py               # Streamlit UI
├── src/
│   ├── __init__.py
│   ├── train.py             # 训练脚本
│   ├── predict.py           # 推理模块
│   ├── rules.py             # 规则模块
│   └── data.py              # 数据工具
├── data/
│   ├── .gitkeep
│   └── ToxiCN_1.0.csv       # (不提交到git)
├── outputs/
│   ├── .gitkeep
│   ├── model/               # (不提交到git)
│   ├── metrics_*.json       # (不提交到git)
│   └── test_predictions.csv # (不提交到git)
├── requirements.txt         # Python 依赖
├── .gitignore              # Git 忽略规则
├── README.md               # 使用文档
├── CONTRIBUTING.md         # 开发指南
├── ARCHITECTURE.md         # 本文件
├── examples.py             # API 示例
└── test_integration.sh     # 集成测试
```

## 技术栈 (Tech Stack)

### 核心框架
- **深度学习**: PyTorch
- **NLP**: Hugging Face Transformers
- **API**: FastAPI + Uvicorn
- **UI**: Streamlit
- **数据处理**: pandas, numpy, scikit-learn

### 模型
- **预训练模型**: `hfl/chinese-roberta-wwm-ext`
- **任务**: Binary Classification (toxic)
- **输出**: Sigmoid 单输出 (0~1)

### 数据集
- **名称**: ToxiCN 1.0
- **许可**: CC BY-NC-ND 4.0
- **用途**: 仅科研非商用

## 性能考虑 (Performance)

### 推理优化
1. **批量处理**: 减少 I/O 开销
2. **GPU 加速**: 自动检测 CUDA
3. **模型缓存**: API 启动时加载一次
4. **规则并行**: 正则匹配高效

### 可扩展性
1. **水平扩展**: 多 worker 部署
2. **负载均衡**: Nginx + Uvicorn
3. **模型版本**: 支持多模型切换
4. **缓存层**: Redis 缓存热点数据

## 安全和合规 (Security & Compliance)

### 数据安全
- ✅ 数据文件不提交到 Git
- ✅ 模型权重不提交到 Git
- ✅ .gitignore 配置完善

### 使用限制
- ⚠️ 仅用于科研和学术用途
- ❌ 禁止商业使用
- ❌ 禁止传播敏感内容

### API 安全建议
- 添加身份认证 (JWT)
- 请求限流 (rate limiting)
- 输入验证和清洗
- 不暴露到公网

## 后续优化方向 (Future Improvements)

1. **模型优化**
   - 模型蒸馏
   - 量化加速
   - ONNX 导出

2. **功能扩展**
   - 多标签分类
   - 解释性分析
   - A/B 测试

3. **工程化**
   - Docker 部署
   - CI/CD 流水线
   - 监控告警

4. **数据增强**
   - 主动学习
   - 半监督学习
   - 对抗样本

---

**版本**: 1.0.0  
**更新日期**: 2026-01-14
