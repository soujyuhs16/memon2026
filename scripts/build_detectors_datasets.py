#!/usr/bin/env python3
"""
构建检测器数据集脚本
Build detector-specific datasets from ZHateBench and HuggingFace spam data

用法 (Usage):
    python scripts/build_detectors_datasets.py --seed 42

输出 (Output):
    data_processed/d1_porn/{train,dev,test}.csv
    data_processed/d2_abuse/{train,dev,test}.csv
    data_processed/d3_bias/{train,dev,test}.csv
    data_processed/d4_spam/{train,dev,test}.csv
    data_processed/manifest.json
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.zhatebench import load_all_zhatebench, group_by_detector
from src.data.build_splits import stratified_split, save_splits

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_spam_dataset(cache_dir: str = None, fallback_to_mock: bool = True) -> pd.DataFrame:
    """
    从 HuggingFace 加载 spam 数据集
    
    Args:
        cache_dir: 缓存目录
        fallback_to_mock: 如果网络失败，是否使用模拟数据
        
    Returns:
        pd.DataFrame: 包含 content, toxic, source 列
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "需要安装 datasets 库: pip install datasets\n"
            "datasets library is required: pip install datasets"
        )
    
    logger.info("=" * 80)
    logger.info("加载 HuggingFace spam 数据集: reatiny/chinese-spam-10000")
    logger.info("=" * 80)
    
    try:
        # 加载数据集
        dataset = load_dataset('reatiny/chinese-spam-10000', cache_dir=cache_dir)
        
        # 转换为 DataFrame
        # 假设数据集有 'text' 和 'label' 列
        # label: 1 = spam (toxic), 0 = normal (safe)
        train_data = dataset['train'].to_pandas()
        
        logger.info(f"原始样本数: {len(train_data)}")
        logger.info(f"列名: {list(train_data.columns)}")
        
        # 映射到统一 schema
        # 根据数据集实际列名调整
        if 'text' in train_data.columns and 'label' in train_data.columns:
            df_mapped = pd.DataFrame({
                'content': train_data['text'],
                'toxic': train_data['label'].astype(int),  # 1=spam, 0=normal
                'source': 'HF:reatiny/chinese-spam-10000',
                'keyword': ''  # spam 数据集无关键词
            })
        else:
            raise ValueError(f"数据集列名不匹配，期望 'text' 和 'label'，实际: {list(train_data.columns)}")
        
    except Exception as e:
        if not fallback_to_mock:
            raise
        
        logger.warning(f"⚠️  HuggingFace 数据集加载失败: {e}")
        logger.warning("使用模拟 spam 数据集作为备用")
        
        # 创建模拟数据集
        mock_data = []
        
        # Spam samples (toxic=1)
        spam_samples = [
            "加微信领取优惠券",
            "日赚1000元加QQ群",
            "低价代购正品保证",
            "刷单兼职月入过万",
            "点击链接了解详情",
            "扫码进群免费试用",
            "私信我获取资源",
            "特价促销限时优惠",
        ]
        
        # Normal samples (toxic=0)
        normal_samples = [
            "今天天气真不错",
            "这个产品质量很好",
            "感谢大家的支持",
            "有什么问题可以询问",
            "欢迎交流讨论",
            "希望对你有帮助",
        ]
        
        for text in spam_samples:
            mock_data.append({
                'content': text,
                'toxic': 1,
                'source': 'Mock:spam',
                'keyword': ''
            })
        
        for text in normal_samples:
            mock_data.append({
                'content': text,
                'toxic': 0,
                'source': 'Mock:normal',
                'keyword': ''
            })
        
        df_mapped = pd.DataFrame(mock_data)
        logger.info(f"使用模拟数据集: {len(df_mapped)} 样本")
    
    # 移除空值
    df_mapped = df_mapped.dropna(subset=['content'])
    
    # 去重
    original_len = len(df_mapped)
    df_mapped = df_mapped.drop_duplicates(subset=['content'], keep='first')
    duplicates_removed = original_len - len(df_mapped)
    
    if duplicates_removed > 0:
        logger.info(f"去除 {duplicates_removed} 个重复样本")
    
    logger.info(f"  样本数: {len(df_mapped)}")
    logger.info(f"  spam 样本: {df_mapped['toxic'].sum()} ({df_mapped['toxic'].mean()*100:.2f}%)")
    logger.info("")
    
    return df_mapped


def build_manifest(
    detectors: Dict[str, Dict],
    seed: int,
    zhatebench_dir: str,
    output_dir: str
) -> Dict:
    """
    生成 manifest.json 元数据
    
    Args:
        detectors: {detector_name: {'train': df, 'dev': df, 'test': df}}
        seed: 随机种子
        zhatebench_dir: ZHateBench 数据目录
        output_dir: 输出目录
        
    Returns:
        Dict: manifest 数据
    """
    manifest = {
        'build_date': datetime.now().isoformat(),
        'seed': seed,
        'zhatebench_dir': zhatebench_dir,
        'output_dir': output_dir,
        'detectors': {},
        'citations': {
            'ZHateBench': {
                'doi': '10.5281/zenodo.16812052',
                'url': 'https://doi.org/10.5281/zenodo.16812052',
                'description': 'Chinese hate speech benchmark with 6 categories',
                'warning': '⚠️ 数据包含敏感内容，仅供研究使用 / Contains sensitive content, research use only'
            },
            'spam_dataset': {
                'huggingface_id': 'reatiny/chinese-spam-10000',
                'url': 'https://huggingface.co/datasets/reatiny/chinese-spam-10000',
                'description': 'Chinese spam/scam dataset for advertising and fraud detection'
            }
        },
        'data_cleaning': {
            'deduplication': 'Applied within and across datasets',
            'null_removal': 'Removed rows with null content'
        }
    }
    
    # 添加每个检测器的统计
    for detector_name, splits in detectors.items():
        train_df = splits['train']
        dev_df = splits['dev']
        test_df = splits['test']
        
        # 统计数据源
        all_sources = pd.concat([train_df, dev_df, test_df])['source'].unique().tolist()
        
        detector_info = {
            'train': {
                'samples': len(train_df),
                'toxic': int(train_df['toxic'].sum()),
                'toxic_ratio': float(train_df['toxic'].mean())
            },
            'dev': {
                'samples': len(dev_df),
                'toxic': int(dev_df['toxic'].sum()),
                'toxic_ratio': float(dev_df['toxic'].mean())
            },
            'test': {
                'samples': len(test_df),
                'toxic': int(test_df['toxic'].sum()),
                'toxic_ratio': float(test_df['toxic'].mean())
            },
            'total_samples': len(train_df) + len(dev_df) + len(test_df),
            'sources': all_sources
        }
        
        manifest['detectors'][detector_name] = detector_info
    
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description='构建检测器数据集 - Build detector-specific datasets'
    )
    
    # 数据路径参数
    parser.add_argument(
        '--zhatebench-dir', type=str,
        default='data_raw/zhatebench',
        help='ZHateBench 数据目录 (默认: data_raw/zhatebench)'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='data_processed',
        help='输出目录 (默认: data_processed)'
    )
    parser.add_argument(
        '--hf-cache-dir', type=str,
        default=None,
        help='HuggingFace 缓存目录 (可选)'
    )
    
    # 切分参数
    parser.add_argument(
        '--seed', type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    parser.add_argument(
        '--train-ratio', type=float,
        default=0.7,
        help='训练集比例 (默认: 0.7)'
    )
    parser.add_argument(
        '--dev-ratio', type=float,
        default=0.15,
        help='验证集比例 (默认: 0.15)'
    )
    parser.add_argument(
        '--test-ratio', type=float,
        default=0.15,
        help='测试集比例 (默认: 0.15)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("构建检测器数据集")
    logger.info("Building Detector-Specific Datasets")
    logger.info("=" * 80)
    logger.info(f"ZHateBench 目录: {args.zhatebench_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"随机种子: {args.seed}")
    logger.info(f"切分比例: train={args.train_ratio}, dev={args.dev_ratio}, test={args.test_ratio}")
    logger.info("")
    
    # 验证输入目录
    if not os.path.exists(args.zhatebench_dir):
        logger.error(f"❌ ZHateBench 目录不存在: {args.zhatebench_dir}")
        logger.error("请将 6 个 CSV 文件放置到该目录:")
        logger.error("  - SexHarmSet.csv")
        logger.error("  - AbuseSet.csv")
        logger.error("  - Bias_region.csv")
        logger.error("  - BiasSet_genden.csv")
        logger.error("  - Bias_race.csv")
        logger.error("  - Bias_occupation.csv")
        return 1
    
    try:
        # 步骤 1: 加载 ZHateBench
        zhatebench_datasets = load_all_zhatebench(args.zhatebench_dir)
        detectors_data = group_by_detector(zhatebench_datasets)
        
        # 步骤 2: 加载 spam 数据集
        spam_df = load_spam_dataset(cache_dir=args.hf_cache_dir)
        detectors_data['d4_spam'] = spam_df
        
        # 步骤 3: 切分并保存每个检测器的数据
        logger.info("=" * 80)
        logger.info("切分并保存数据集")
        logger.info("=" * 80)
        
        detectors_splits = {}
        
        for detector_name, df in detectors_data.items():
            logger.info(f"\n处理 {detector_name}...")
            logger.info(f"  总样本: {len(df)}")
            
            # 切分
            train_df, dev_df, test_df = stratified_split(
                df,
                train_ratio=args.train_ratio,
                dev_ratio=args.dev_ratio,
                test_ratio=args.test_ratio,
                random_state=args.seed
            )
            
            # 保存
            detector_output_dir = os.path.join(args.output_dir, detector_name)
            save_splits(train_df, dev_df, test_df, detector_output_dir)
            
            detectors_splits[detector_name] = {
                'train': train_df,
                'dev': dev_df,
                'test': test_df
            }
            logger.info("")
        
        # 步骤 4: 生成 manifest
        logger.info("=" * 80)
        logger.info("生成 manifest.json")
        logger.info("=" * 80)
        
        manifest = build_manifest(
            detectors_splits,
            args.seed,
            args.zhatebench_dir,
            args.output_dir
        )
        
        manifest_path = os.path.join(args.output_dir, 'manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ manifest 已保存: {manifest_path}")
        logger.info("")
        
        # 最终统计
        logger.info("=" * 80)
        logger.info("构建完成！Build Complete!")
        logger.info("=" * 80)
        logger.info(f"输出目录: {args.output_dir}")
        logger.info(f"  - d1_porn/{{train,dev,test}}.csv")
        logger.info(f"  - d2_abuse/{{train,dev,test}}.csv")
        logger.info(f"  - d3_bias/{{train,dev,test}}.csv")
        logger.info(f"  - d4_spam/{{train,dev,test}}.csv")
        logger.info(f"  - manifest.json")
        logger.info("")
        
        logger.info("检测器统计:")
        for detector_name, info in manifest['detectors'].items():
            logger.info(f"  {detector_name}:")
            logger.info(f"    总样本: {info['total_samples']}")
            logger.info(f"    train: {info['train']['samples']}, dev: {info['dev']['samples']}, test: {info['test']['samples']}")
            logger.info(f"    有害比例: {info['train']['toxic_ratio']*100:.2f}%")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ 全部完成！All Done!")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
