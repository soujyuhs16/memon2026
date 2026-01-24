#!/usr/bin/env python3
"""
验证 category_hint 功能的单元测试
Unit tests for category_hint functionality
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_abuse_keywords_loading():
    """测试辱骂关键词加载"""
    from src.predict import load_abuse_keywords
    
    keywords = load_abuse_keywords()
    assert len(keywords) > 0, "Should load at least some keywords"
    assert '傻逼' in keywords or '垃圾' in keywords, "Should contain expected keywords"
    print("✅ test_abuse_keywords_loading passed")


def test_check_abuse_keywords():
    """测试辱骂关键词检测"""
    from src.predict import check_abuse_keywords
    
    # Test positive cases
    assert check_abuse_keywords("你这个傻逼") == True, "Should detect abuse"
    assert check_abuse_keywords("真是垃圾") == True, "Should detect abuse"
    
    # Test negative cases  
    assert check_abuse_keywords("这个产品很好用") == False, "Should not detect abuse"
    assert check_abuse_keywords("加微信VX123") == False, "Should not detect abuse"
    
    print("✅ test_check_abuse_keywords passed")


def test_determine_category_hint():
    """测试类别提示判定逻辑"""
    from src.predict import determine_category_hint
    
    # Test case 1: Safe content (pred=0)
    hint = determine_category_hint("正常文本", [], 0)
    assert hint == "", f"Safe content should have empty hint, got: {hint}"
    
    # Test case 2: Ad/spam (rule hits)
    hint = determine_category_hint("加微信VX", ["WeChat"], 1)
    assert hint == "广告/引流", f"Rule hits should return '广告/引流', got: {hint}"
    
    # Test case 3: Abuse (keyword match)
    hint = determine_category_hint("你是傻逼", [], 1)
    assert hint == "辱骂/仇恨/攻击", f"Abuse keywords should return '辱骂/仇恨/攻击', got: {hint}"
    
    # Test case 4: Model only (no rules, no keywords)
    hint = determine_category_hint("某种问题内容", [], 1)
    assert hint == "模型判定（未命中规则/词表）", f"Model only should return correct hint, got: {hint}"
    
    print("✅ test_determine_category_hint passed")


def test_api_response_model():
    """测试 API 响应模型包含 category_hint"""
    from api.main import PredictResponse
    
    # Check that PredictResponse has category_hint field
    assert hasattr(PredictResponse, 'model_fields') or hasattr(PredictResponse, '__fields__'), \
        "PredictResponse should be a Pydantic model"
    
    # Get fields
    if hasattr(PredictResponse, 'model_fields'):
        fields = PredictResponse.model_fields
    else:
        fields = PredictResponse.__fields__
    
    assert 'category_hint' in fields, "PredictResponse should have category_hint field"
    print("✅ test_api_response_model passed")


def main():
    print("="*60)
    print("运行 category_hint 功能验证测试")
    print("="*60)
    print()
    
    try:
        test_abuse_keywords_loading()
        test_check_abuse_keywords()
        test_determine_category_hint()
        test_api_response_model()
        
        print()
        print("="*60)
        print("所有测试通过！✅")
        print("="*60)
        return 0
        
    except AssertionError as e:
        print()
        print("="*60)
        print(f"测试失败！❌")
        print(f"错误: {e}")
        print("="*60)
        return 1
    except Exception as e:
        print()
        print("="*60)
        print(f"测试出错！❌")
        print(f"错误: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
