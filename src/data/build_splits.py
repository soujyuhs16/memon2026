"""
数据集切分工具
Stratified train/dev/test splitting utilities
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    分层切分数据集为 train/dev/test
    
    Args:
        df: 输入数据框，必须包含 'toxic' 列
        train_ratio: 训练集比例
        dev_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子（确保可复现）
        
    Returns:
        Tuple[train_df, dev_df, test_df]
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
        f"比例之和必须为 1.0: {train_ratio} + {dev_ratio} + {test_ratio} = {train_ratio + dev_ratio + test_ratio}"
    
    if 'toxic' not in df.columns:
        raise ValueError("数据框必须包含 'toxic' 列用于分层抽样")
    
    # 检查每个类别的样本数
    toxic_counts = df['toxic'].value_counts()
    logger.info(f"数据分布: toxic=0: {toxic_counts.get(0, 0)}, toxic=1: {toxic_counts.get(1, 0)}")
    
    # 第一步：切分出测试集
    train_dev_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=df['toxic']
    )
    
    # 第二步：从 train_dev 中切分出验证集
    # dev 占 train_dev 的比例
    dev_ratio_adjusted = dev_ratio / (train_ratio + dev_ratio)
    
    train_df, dev_df = train_test_split(
        train_dev_df,
        test_size=dev_ratio_adjusted,
        random_state=random_state,
        stratify=train_dev_df['toxic']
    )
    
    logger.info(f"数据切分完成:")
    logger.info(f"  训练集: {len(train_df)} 样本 (toxic={train_df['toxic'].sum()}, {train_df['toxic'].mean()*100:.2f}%)")
    logger.info(f"  验证集: {len(dev_df)} 样本 (toxic={dev_df['toxic'].sum()}, {dev_df['toxic'].mean()*100:.2f}%)")
    logger.info(f"  测试集: {len(test_df)} 样本 (toxic={test_df['toxic'].sum()}, {test_df['toxic'].mean()*100:.2f}%)")
    
    return train_df, dev_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    prefix: str = ""
):
    """
    保存 train/dev/test 到 CSV 文件
    
    Args:
        train_df: 训练集
        dev_df: 验证集
        test_df: 测试集
        output_dir: 输出目录
        prefix: 文件名前缀（可选）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建文件名
    if prefix:
        train_file = os.path.join(output_dir, f"{prefix}_train.csv")
        dev_file = os.path.join(output_dir, f"{prefix}_dev.csv")
        test_file = os.path.join(output_dir, f"{prefix}_test.csv")
    else:
        train_file = os.path.join(output_dir, "train.csv")
        dev_file = os.path.join(output_dir, "dev.csv")
        test_file = os.path.join(output_dir, "test.csv")
    
    # 保存
    train_df.to_csv(train_file, index=False, encoding='utf-8')
    dev_df.to_csv(dev_file, index=False, encoding='utf-8')
    test_df.to_csv(test_file, index=False, encoding='utf-8')
    
    logger.info(f"✅ 已保存到 {output_dir}:")
    logger.info(f"  - {os.path.basename(train_file)}")
    logger.info(f"  - {os.path.basename(dev_file)}")
    logger.info(f"  - {os.path.basename(test_file)}")


if __name__ == '__main__':
    # 测试用例
    import numpy as np
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'content': [f'sample_{i}' for i in range(n_samples)],
        'toxic': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'source': 'test',
        'keyword': 'test_keyword'
    })
    
    print("示例数据:")
    print(df.head())
    print(f"\n总样本: {len(df)}")
    print(f"有害样本: {df['toxic'].sum()} ({df['toxic'].mean()*100:.2f}%)")
    print()
    
    # 测试切分
    train_df, dev_df, test_df = stratified_split(df, random_state=42)
    
    print("\n✅ 切分测试通过")
