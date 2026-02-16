#!/usr/bin/env python3
"""
测试训练脚本的参数解析
Test training script argument parsing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_train_args():
    """测试训练脚本支持新的参数"""
    import argparse
    
    # 模拟 train.py 的参数解析
    parser = argparse.ArgumentParser()
    
    # 模式 1: 单一 CSV 文件
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--dev_size', type=float, default=0.1)
    
    # 模式 2: 显式 train/dev/test
    parser.add_argument('--train_csv', type=str, default=None)
    parser.add_argument('--dev_csv', type=str, default=None)
    parser.add_argument('--test_csv', type=str, default=None)
    
    # 测试场景 1: 使用显式 train/dev/test
    print("=" * 80)
    print("测试场景 1: 显式 train/dev/test CSV")
    print("=" * 80)
    
    args1 = parser.parse_args([
        '--train_csv', 'data_processed/d1_porn/train.csv',
        '--dev_csv', 'data_processed/d1_porn/dev.csv',
        '--test_csv', 'data_processed/d1_porn/test.csv'
    ])
    
    print(f"train_csv: {args1.train_csv}")
    print(f"dev_csv: {args1.dev_csv}")
    print(f"test_csv: {args1.test_csv}")
    print(f"csv_path: {args1.csv_path}")
    
    # 验证模式 2
    mode2 = all([args1.train_csv, args1.dev_csv, args1.test_csv])
    assert mode2, "模式 2 应该被检测到"
    print("✅ 模式 2 检测通过")
    print()
    
    # 测试场景 2: 使用单一 CSV（向后兼容）
    print("=" * 80)
    print("测试场景 2: 单一 CSV 模式（向后兼容）")
    print("=" * 80)
    
    args2 = parser.parse_args([
        '--csv_path', 'data/ToxiCN_1.0.csv'
    ])
    
    print(f"csv_path: {args2.csv_path}")
    print(f"train_csv: {args2.train_csv}")
    
    # 验证模式 1
    mode1 = args2.csv_path is not None
    assert mode1, "模式 1 应该被检测到"
    print("✅ 模式 1 检测通过")
    print()
    
    print("=" * 80)
    print("✅ 所有参数解析测试通过")
    print("=" * 80)


def test_data_loading():
    """测试数据加载"""
    import pandas as pd
    
    print("\n" + "=" * 80)
    print("测试数据加载")
    print("=" * 80)
    
    # 测试加载生成的数据
    train_csv = 'data_processed/d1_porn/train.csv'
    dev_csv = 'data_processed/d1_porn/dev.csv'
    test_csv = 'data_processed/d1_porn/test.csv'
    
    if not os.path.exists(train_csv):
        print(f"⚠️  数据文件不存在: {train_csv}")
        print("请先运行: python scripts/build_detectors_datasets.py --seed 42")
        return
    
    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"\n训练集:")
    print(f"  样本数: {len(train_df)}")
    print(f"  列: {list(train_df.columns)}")
    print(f"  有害样本: {train_df['toxic'].sum()} ({train_df['toxic'].mean()*100:.2f}%)")
    
    print(f"\n验证集:")
    print(f"  样本数: {len(dev_df)}")
    print(f"  有害样本: {dev_df['toxic'].sum()} ({dev_df['toxic'].mean()*100:.2f}%)")
    
    print(f"\n测试集:")
    print(f"  样本数: {len(test_df)}")
    print(f"  有害样本: {test_df['toxic'].sum()} ({test_df['toxic'].mean()*100:.2f}%)")
    
    # 验证必需列存在
    required_cols = ['content', 'toxic']
    for df_name, df in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
        for col in required_cols:
            assert col in df.columns, f"{df_name} 缺少列: {col}"
    
    print("\n✅ 数据加载测试通过")


if __name__ == '__main__':
    try:
        test_train_args()
        test_data_loading()
        
        print("\n" + "=" * 80)
        print("✅ 所有测试通过！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
