"""
ZHateBench 数据加载模块
Load and process ZHateBench dataset CSVs

ZHateBench 引用 / Citation:
DOI: 10.5281/zenodo.16812052
https://doi.org/10.5281/zenodo.16812052

⚠️ 数据仅供研究使用 / Research use only
包含敏感内容 / Contains sensitive content
"""

import os
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


# ZHateBench 6个数据集文件名
ZHATEBENCH_FILES = {
    'SexHarmSet': 'SexHarmSet.csv',           # D1: 涉黄
    'AbuseSet': 'AbuseSet.csv',               # D2: 辱骂
    'Bias_region': 'Bias_region.csv',         # D3: 地域歧视
    'BiasSet_genden': 'BiasSet_genden.csv',   # D3: 性别歧视
    'Bias_race': 'Bias_race.csv',             # D3: 种族歧视
    'Bias_occupation': 'Bias_occupation.csv'  # D3: 职业歧视
}

# 映射到检测器
DETECTOR_MAPPING = {
    'SexHarmSet': 'd1_porn',
    'AbuseSet': 'd2_abuse',
    'Bias_region': 'd3_bias',
    'BiasSet_genden': 'd3_bias',
    'Bias_race': 'd3_bias',
    'Bias_occupation': 'd3_bias'
}


def load_zhatebench_csv(csv_path: str, dataset_name: str) -> pd.DataFrame:
    """
    加载单个 ZHateBench CSV 文件
    
    Args:
        csv_path: CSV 文件路径
        dataset_name: 数据集名称 (如 'SexHarmSet')
        
    Returns:
        pd.DataFrame: 包含 content, toxic, source, keyword 列
        
    Expected CSV columns: Keyword, Type, Sentence
    - Type: 'Harmful' or 'Safe'
    - Sentence: 文本内容
    - Keyword: 关键词（可选，用于可解释性）
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ZHateBench 文件不存在: {csv_path}")
    
    # 读取 CSV
    df = pd.read_csv(csv_path)
    
    # 验证列
    required_cols = ['Keyword', 'Type', 'Sentence']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 文件缺少必需列: {missing_cols}. 当前列: {list(df.columns)}")
    
    # 映射到统一 schema
    df_mapped = pd.DataFrame({
        'content': df['Sentence'],
        'toxic': (df['Type'] == 'Harmful').astype(int),  # Harmful -> 1, Safe -> 0
        'source': f'ZHateBench:{dataset_name}',
        'keyword': df['Keyword']
    })
    
    # 移除空值
    df_mapped = df_mapped.dropna(subset=['content'])
    
    # 去重（基于 content）
    original_len = len(df_mapped)
    df_mapped = df_mapped.drop_duplicates(subset=['content'], keep='first')
    duplicates_removed = original_len - len(df_mapped)
    
    if duplicates_removed > 0:
        logger.info(f"{dataset_name}: 去除 {duplicates_removed} 个重复样本")
    
    return df_mapped


def load_all_zhatebench(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    加载所有 ZHateBench 数据集
    
    Args:
        data_dir: ZHateBench 数据目录 (如 'data_raw/zhatebench/')
        
    Returns:
        Dict[str, pd.DataFrame]: {dataset_name: dataframe}
    """
    datasets = {}
    
    logger.info("=" * 80)
    logger.info("加载 ZHateBench 数据集")
    logger.info("=" * 80)
    
    for dataset_name, filename in ZHATEBENCH_FILES.items():
        csv_path = os.path.join(data_dir, filename)
        logger.info(f"加载 {dataset_name} from {csv_path}")
        
        try:
            df = load_zhatebench_csv(csv_path, dataset_name)
            datasets[dataset_name] = df
            
            logger.info(f"  样本数: {len(df)}")
            logger.info(f"  有害样本: {df['toxic'].sum()} ({df['toxic'].mean()*100:.2f}%)")
            logger.info("")
            
        except FileNotFoundError as e:
            logger.error(f"  ❌ 文件未找到: {e}")
            raise
        except Exception as e:
            logger.error(f"  ❌ 加载失败: {e}")
            raise
    
    # 统计
    total_samples = sum(len(df) for df in datasets.values())
    total_harmful = sum(df['toxic'].sum() for df in datasets.values())
    
    logger.info("=" * 80)
    logger.info(f"✅ ZHateBench 加载完成: {len(datasets)} 个数据集, {total_samples} 个样本")
    logger.info(f"   有害样本总数: {total_harmful} ({total_harmful/total_samples*100:.2f}%)")
    logger.info("=" * 80)
    logger.info("")
    
    return datasets


def group_by_detector(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    将 ZHateBench 数据集按检测器分组
    
    Args:
        datasets: {dataset_name: dataframe}
        
    Returns:
        Dict[str, pd.DataFrame]: {detector_name: merged_dataframe}
        例如: {'d1_porn': df1, 'd2_abuse': df2, 'd3_bias': df3}
    """
    detectors = {}
    
    for dataset_name, df in datasets.items():
        detector = DETECTOR_MAPPING[dataset_name]
        
        if detector not in detectors:
            detectors[detector] = []
        
        detectors[detector].append(df)
    
    # 合并同一检测器的数据
    merged_detectors = {}
    for detector, dfs in detectors.items():
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # 再次去重（跨数据集）
        original_len = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['content'], keep='first')
        duplicates_removed = original_len - len(merged_df)
        
        if duplicates_removed > 0:
            logger.info(f"{detector}: 跨数据集去重，移除 {duplicates_removed} 个样本")
        
        merged_detectors[detector] = merged_df
    
    return merged_detectors


if __name__ == '__main__':
    # 测试用例
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # 测试加载
    data_dir = 'data_raw/zhatebench/'
    
    if not os.path.exists(data_dir):
        print(f"测试数据目录不存在: {data_dir}")
        print("请将 ZHateBench CSV 文件放置到该目录")
        sys.exit(1)
    
    try:
        datasets = load_all_zhatebench(data_dir)
        detectors = group_by_detector(datasets)
        
        print("\n按检测器分组统计:")
        print("=" * 80)
        for detector, df in detectors.items():
            print(f"{detector}:")
            print(f"  总样本: {len(df)}")
            print(f"  有害样本: {df['toxic'].sum()} ({df['toxic'].mean()*100:.2f}%)")
            print(f"  数据源: {df['source'].unique()}")
            print()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)
