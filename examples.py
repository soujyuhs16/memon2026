"""
API 使用示例
Example API usage demonstrations
"""

# ============================================================================
# 示例 1: 使用 requests 库调用 API
# Example 1: Using requests library
# ============================================================================

print("=" * 80)
print("API 使用示例")
print("=" * 80)
print()

# 单条预测示例
print("1. 单条预测示例:")
print("-" * 80)
print("""
import requests

# 正常评论
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "这个产品很好用，质量不错",
        "threshold": 0.5
    }
)
result = response.json()
print(f"文本: {result['text']}")
print(f"模型概率: {result['model_prob']:.3f}")
print(f"最终概率: {result['final_prob']:.3f}")
print(f"预测: {'有毒' if result['pred'] == 1 else '正常'}")
print(f"规则命中: {result['rule_hits']}")
""")
print()

# 广告检测示例
print("2. 广告检测示例:")
print("-" * 80)
print("""
# 广告导流评论
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "加微信VX123456了解详情，低价批发返现",
        "threshold": 0.5
    }
)
result = response.json()
print(f"文本: {result['text']}")
print(f"模型概率: {result['model_prob']:.3f}")
print(f"规则分数: {result['rule_score']:.3f}")
print(f"最终概率: {result['final_prob']:.3f}")
print(f"预测: {'有毒/广告' if result['pred'] == 1 else '正常'}")
print(f"规则命中: {result['rule_hits']}")
# 输出: rule_hits = ['WeChat', 'Price']
""")
print()

# 批量预测示例
print("3. 批量CSV预测示例:")
print("-" * 80)
print("""
import pandas as pd
import requests

# 准备测试数据
test_data = pd.DataFrame({
    'content': [
        '这个产品很好用',
        '加微信了解详情',
        '质量不错推荐',
        'QQ群123456789低价批发'
    ]
})

# 保存为CSV
test_data.to_csv('test_comments.csv', index=False)

# 批量预测
with open('test_comments.csv', 'rb') as f:
    response = requests.post(
        "http://localhost:8000/batch_predict",
        files={'file': f},
        data={
            'threshold': 0.5,
            'text_column': 'content'
        }
    )

# 保存结果
with open('predictions.csv', 'wb') as f:
    f.write(response.content)

# 读取结果
results = pd.read_csv('predictions.csv')
print(results)
""")
print()

# curl 命令示例
print("4. curl 命令示例:")
print("-" * 80)
print("""
# 单条预测
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "这个产品很好用",
    "threshold": 0.5
  }'

# 批量预测
curl -X POST "http://localhost:8000/batch_predict" \\
  -F "file=@test_comments.csv" \\
  -F "threshold=0.5" \\
  -F "text_column=content" \\
  -o predictions.csv

# 健康检查
curl "http://localhost:8000/health"
""")
print()

# ============================================================================
# 示例 2: 直接使用 Python 模块
# Example 2: Direct Python module usage
# ============================================================================

print("5. 直接使用 Python 模块:")
print("-" * 80)
print("""
from src.predict import load_model

# 加载模型
classifier = load_model('outputs/model')

# 单条预测
result = classifier.predict_single(
    text="这是一条测试评论",
    threshold=0.5,
    use_rules=True
)
print(result)

# 批量预测
texts = [
    "这个产品很好用",
    "加微信了解详情",
    "质量不错推荐"
]
results = classifier.predict_batch(texts, threshold=0.5)
for r in results:
    print(f"{r['text']}: {r['pred']}")
""")
print()

# ============================================================================
# 示例 3: 规则检测单独使用
# Example 3: Standalone rule detection
# ============================================================================

print("6. 单独使用规则检测:")
print("-" * 80)
print("""
from src.rules import check_rules

# 检测广告
text = "加微信VX123456了解详情，QQ群123456789"
result = check_rules(text)

print(f"文本: {text}")
print(f"命中规则: {result['hits']}")
print(f"规则分数: {result['score']}")
# 输出: hits = ['WeChat', 'QQ'], score = 1.0
""")
print()

print("=" * 80)
print("更多示例请参考 README.md")
print("=" * 80)
