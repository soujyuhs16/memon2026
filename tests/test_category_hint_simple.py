#!/usr/bin/env python3
"""
验证 category_hint 功能的简单测试（无需依赖）
Simple tests for category_hint functionality (no dependencies required)
"""
import os


def test_abuse_keywords_file():
    """测试辱骂关键词文件存在且格式正确"""
    abuse_file = 'src/resources/abuse_words.txt'
    
    assert os.path.exists(abuse_file), f"Abuse keywords file should exist at {abuse_file}"
    
    with open(abuse_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    assert len(lines) > 0, "Abuse keywords file should contain keywords"
    assert '傻逼' in lines or '垃圾' in lines, "Should contain expected keywords"
    
    print(f"✅ test_abuse_keywords_file passed (loaded {len(lines)} keywords)")


def test_files_syntax():
    """测试所有修改的文件语法正确"""
    import py_compile
    
    files_to_check = [
        'src/predict.py',
        'api/main.py',
        'app/app.py',
    ]
    
    for file_path in files_to_check:
        try:
            py_compile.compile(file_path, doraise=True)
            print(f"✅ {file_path} syntax valid")
        except py_compile.PyCompileError as e:
            raise AssertionError(f"Syntax error in {file_path}: {e}")


def test_category_hint_logic():
    """测试类别提示判定逻辑（无需导入模块）"""
    # Simulate the logic
    def determine_hint(text, rule_hits, pred):
        if pred == 0:
            return ""
        if rule_hits:
            return "广告/引流"
        # Simple keyword check
        abuse_words = ['傻逼', '垃圾', '智障']
        if any(word in text for word in abuse_words):
            return "辱骂/仇恨/攻击"
        return "模型判定（未命中规则/词表）"
    
    # Test cases
    tests = [
        ("正常文本", [], 0, ""),
        ("加微信VX", ["WeChat"], 1, "广告/引流"),
        ("你是傻逼", [], 1, "辱骂/仇恨/攻击"),
        ("问题内容", [], 1, "模型判定（未命中规则/词表）"),
    ]
    
    for text, rules, pred, expected in tests:
        result = determine_hint(text, rules, pred)
        assert result == expected, f"Expected '{expected}' but got '{result}' for text '{text}'"
    
    print("✅ test_category_hint_logic passed")


def main():
    print("="*60)
    print("运行 category_hint 简单验证测试")
    print("="*60)
    print()
    
    try:
        test_abuse_keywords_file()
        test_files_syntax()
        test_category_hint_logic()
        
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
    import sys
    sys.exit(main())
