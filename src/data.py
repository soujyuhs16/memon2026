"""
数据加载与预处理模块
Data loading and preprocessing utilities
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import os


def load_toxicn_data(csv_path: str) -> pd.DataFrame:
    """
    加载 ToxiCN 数据集
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        pd.DataFrame: 包含 content 和 toxic 列的数据框
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # 确保必要的列存在
    if 'content' not in df.columns or 'toxic' not in df.columns:
        raise ValueError("数据集必须包含 'content' 和 'toxic' 列")
    
    # 移除空值
    df = df.dropna(subset=['content', 'toxic'])
    
    # 确保 toxic 列是 0/1 整数
    df['toxic'] = df['toxic'].astype(int)
    
    return df


def split_data(df: pd.DataFrame, 
               test_size: float = 0.2, 
               dev_size: float = 0.1,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    划分训练集、验证集、测试集
    
    Args:
        df: 原始数据框
        test_size: 测试集比例
        dev_size: 验证集比例（从训练集中切分）
        random_state: 随机种子
        
    Returns:
        Tuple[train_df, dev_df, test_df]
    """
    # 先切分出测试集（分层抽样）
    train_dev_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['toxic']
    )
    
    # 从训练集中切分验证集
    train_df, dev_df = train_test_split(
        train_dev_df,
        test_size=dev_size,
        random_state=random_state,
        stratify=train_dev_df['toxic']
    )
    
    print(f"数据划分完成:")
    print(f"  训练集: {len(train_df)} 样本 (toxic={train_df['toxic'].sum()})")
    print(f"  验证集: {len(dev_df)} 样本 (toxic={dev_df['toxic'].sum()})")
    print(f"  测试集: {len(test_df)} 样本 (toxic={test_df['toxic'].sum()})")
    
    return train_df, dev_df, test_df


def prepare_dataset_for_training(df: pd.DataFrame) -> Dict:
    """
    将 DataFrame 转换为适合 Transformers 的格式
    
    Args:
        df: 包含 content 和 toxic 列的数据框
        
    Returns:
        Dict: {'text': list, 'labels': list (float32)}
    """
    # 验证 toxic 列的值
    if not df['toxic'].isin([0, 1]).all():
        raise ValueError("toxic 列应只包含 0 或 1 的值")
    
    return {
        'text': df['content'].tolist(),
        'labels': df['toxic'].astype('float32').tolist()
    }


if __name__ == '__main__':
    # 测试用例
    print("数据加载模块测试")
    print("=" * 60)
    
    # 创建示例数据
    sample_data = pd.DataFrame({
        'content': [
            '这个产品很好用',
            '你个傻X',
            '服务态度很好',
            '垃圾东西，骗子',
            '推荐大家购买',
        ],
        'toxic': [0, 1, 0, 1, 0]
    })
    
    print("\n示例数据:")
    print(sample_data)
    
    train_df, dev_df, test_df = split_data(sample_data, test_size=0.2, dev_size=0.2)
    print(f"\n训练集样本数: {len(train_df)}")
    print(f"验证集样本数: {len(dev_df)}")
    print(f"测试集样本数: {len(test_df)}")
