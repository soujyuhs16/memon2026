"""
广告导流规则检测模块
Advertising/spam detection rules module
"""
import re
from typing import Dict, List, Tuple


# 规则融合配置
RULE_OVERRIDE = False  # 如果为 True，规则命中直接判定为有毒
DEFAULT_RULE_WEIGHT = 1.0  # 规则得分权重


def check_rules(text: str) -> Dict:
    """
    检查文本是否命中广告导流规则
    
    Args:
        text: 输入文本
        
    Returns:
        dict: {
            'hits': list[str],  # 命中的规则列表
            'score': float      # 0~1 分数，命中越多分数越高
        }
    """
    hits = []
    
    # 规则1: URL检测（http/https/www）
    if re.search(r'(https?://|www\.)\S+', text, re.IGNORECASE):
        hits.append('URL')
    
    # 规则2: 微信/VX/wx检测
    if re.search(r'(微信|VX|vx|Vx|wx|WX|Wx|威信)', text):
        hits.append('WeChat')
    
    # 规则3: QQ号检测（5-11位数字）
    if re.search(r'(QQ|qq|Qq|扣扣|q群)\s*[:：]?\s*\d{5,11}', text):
        hits.append('QQ')
    
    # 规则4: 手机号检测（1开头11位）
    if re.search(r'1[3-9]\d{9}', text):
        hits.append('Phone')
    
    # 规则5: 价格/优惠/返现关键词
    if re.search(r'(低价|优惠|返现|返利|代购|批发|折扣|特价|仅需|只需|\d+元)', text):
        hits.append('Price')
    
    # 规则6: 加群/进群关键词
    if re.search(r'(加群|进群|入群|群号|拉群|扫码|二维码)', text):
        hits.append('Group')
    
    # 规则7: 刷单/兼职等常见spam
    if re.search(r'(刷单|兼职|招聘|日赚|月入|小时赚|躺赚)', text):
        hits.append('Scam')
    
    # 规则8: 私信/联系方式引导
    if re.search(r'(私信|私聊|联系|详情|咨询|了解)', text):
        hits.append('Contact')
    
    # 计算分数：命中规则越多，分数越高
    # 简单策略：命中1个规则 = 0.5分，2个及以上 = 1.0分
    if len(hits) == 0:
        score = 0.0
    elif len(hits) == 1:
        score = 0.6
    else:
        score = 1.0
    
    return {
        'hits': hits,
        'score': score
    }


def merge_predictions(model_prob: float, rule_score: float, 
                      rule_override: bool = RULE_OVERRIDE) -> float:
    """
    融合模型预测和规则得分
    
    Args:
        model_prob: 模型预测概率 (0~1)
        rule_score: 规则得分 (0~1)
        rule_override: 是否规则直接覆盖
        
    Returns:
        float: 最终概率
    """
    if rule_override and rule_score > 0:
        return 1.0
    else:
        # 默认策略：取最大值
        return max(model_prob, rule_score)


if __name__ == '__main__':
    # 测试用例
    test_cases = [
        "这个产品很好用",
        "加微信VX123456了解详情",
        "QQ群：123456789，低价批发",
        "请访问 http://example.com 获取优惠",
        "私信我，手机号13812345678"
    ]
    
    print("规则检测测试：")
    print("=" * 60)
    for text in test_cases:
        result = check_rules(text)
        print(f"\n文本: {text}")
        print(f"命中规则: {result['hits']}")
        print(f"规则分数: {result['score']:.2f}")
