# Category Hint 功能实现总结

## 📝 变更概述

本 PR 实现了在 Streamlit 和 FastAPI 推理输出中增加 `category_hint`（类别提示）字段，并将界面文案统一为"广义有害内容（辱骂/仇恨/引流广告）"。

## 🎯 主要变更

### 1. 新增辱骂关键词词表
- **文件**: `src/resources/abuse_words.txt`
- **内容**: 包含 36+ 个辱骂/仇恨/攻击类关键词
- **用途**: 用于识别辱骂类有害内容

### 2. 推理模块增强 (`src/predict.py`)
- 新增 `load_abuse_keywords()`: 加载辱骂关键词
- 新增 `check_abuse_keywords()`: 检测文本是否包含辱骂关键词
- 新增 `determine_category_hint()`: 确定内容类别提示
- 更新 `predict_one()` 和 `predict_batch()`: 返回结果中增加 `category_hint` 字段

### 3. Streamlit UI 更新 (`app/app.py`)
- 警告文案：从"有毒/广告"更新为"有害内容（辱骂/仇恨/引流广告）"
- 判定结果显示：
  - pred=1: "🚫 有害内容（辱骂/仇恨/引流广告）"
  - pred=0: "✅ 安全内容"
- 新增类别提示展示：在判定结果下方显示 `category_hint`
- 统计指标：从"有毒/广告"更新为"有害内容"

### 4. FastAPI 更新 (`api/main.py`)
- `PredictResponse` 模型增加 `category_hint: str` 字段
- 批量预测 CSV 输出包含 `category_hint` 列
- API 描述更新为"有害内容分类服务（辱骂/仇恨/引流广告）"

### 5. 文档更新 (`README.md`)
- 系统特性说明中增加类别提示功能描述
- 新增"类别提示"专门章节说明三种类型
- API 示例中增加 `category_hint` 字段展示
- 新增辱骂检测示例
- 输出格式章节完整描述所有字段

## 🔍 Category Hint 逻辑

```
if pred == 0 (安全内容):
    category_hint = ""
    
elif rule_hits (规则命中):
    category_hint = "广告/引流"
    
elif check_abuse_keywords(text) (辱骂关键词命中):
    category_hint = "辱骂/仇恨/攻击"
    
else (仅模型判定):
    category_hint = "模型判定（未命中规则/词表）"
```

## 📊 输出示例

### 示例 1: 安全内容
```json
{
  "text": "这个产品很好用",
  "pred": 0,
  "category_hint": ""
}
```

### 示例 2: 广告/引流
```json
{
  "text": "加微信VX123了解详情",
  "pred": 1,
  "rule_hits": ["WeChat"],
  "category_hint": "广告/引流"
}
```

### 示例 3: 辱骂/仇恨/攻击
```json
{
  "text": "你这个傻逼",
  "pred": 1,
  "rule_hits": [],
  "category_hint": "辱骂/仇恨/攻击"
}
```

### 示例 4: 模型判定
```json
{
  "text": "某种其他有问题的内容",
  "pred": 1,
  "rule_hits": [],
  "category_hint": "模型判定（未命中规则/词表）"
}
```

## 🎨 Streamlit UI 变化

### 单条预测页面
**原来:**
```
判定结果: 🚫 有毒/广告
```

**现在:**
```
判定结果: 🚫 有害内容（辱骂/仇恨/引流广告）
可能类别: 广告/引流  [显示为信息框]
```

### 批量预测统计
**原来:**
```
总样本数: 100
有毒/广告: 25
正常: 75
```

**现在:**
```
总样本数: 100
有害内容: 25
安全内容: 75
```

### 输出 CSV 列
原有列 + 新增 `category_hint` 列

## ✅ 测试验证

### 单元测试
- `tests/test_category_hint_simple.py`: 验证核心逻辑（无依赖）
- `tests/test_category_hint.py`: 完整单元测试（需要依赖）

### 集成测试
运行 `bash test_integration.sh` 验证所有模块语法正确

### 验证清单
- ✅ 辱骂关键词文件加载正常（36 个关键词）
- ✅ 类别提示判定逻辑正确（4 种情况）
- ✅ API 响应模型包含 category_hint 字段
- ✅ 所有 Python 文件语法正确
- ✅ 集成测试通过

## 🔧 兼容性

### 向后兼容
- ✅ 所有现有字段保持不变
- ✅ 仅新增 `category_hint` 字段
- ✅ 现有训练和推理命令不受影响
- ✅ API 响应增加字段，向前兼容

### 依赖要求
无新增依赖，使用现有环境即可运行

## 📈 预期效果

1. **更准确的内容分类**: 用户可以快速了解有害内容的大致类型
2. **更好的用户体验**: UI 文案更加清晰和专业
3. **便于审核**: 审核人员可以根据类别提示快速决策
4. **数据分析**: 可以统计不同类型有害内容的比例

## ⚠️ 注意事项

1. `category_hint` 是启发式规则，仅供参考，不代表精确分类
2. 辱骂关键词词表需要根据实际情况定期维护更新
3. 规则优先于关键词检测，符合实际业务逻辑
4. 仅当 pred=1 时 category_hint 才有非空值

## 📁 修改文件清单

```
src/resources/abuse_words.txt  (新增)
src/predict.py                 (修改)
app/app.py                     (修改)
api/main.py                    (修改)
README.md                      (修改)
tests/test_category_hint.py   (新增)
tests/test_category_hint_simple.py (新增)
```

## 🎉 完成度

所有需求已实现并通过测试：
- ✅ 创建辱骂关键词词表
- ✅ 实现 category_hint 判定逻辑
- ✅ 更新推理模块
- ✅ 更新 Streamlit UI
- ✅ 更新 FastAPI 接口
- ✅ 更新文档
- ✅ 添加单元测试
