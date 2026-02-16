#!/usr/bin/env python3
"""
数据清洗与构建脚本
Data cleaning and building script

功能 (Features):
1. 检测 CHSD toxic=0 样本中的"明显有害"内容
2. 默认策略：剔除（drop）命中样本
3. 可选策略：对高置信命中翻标为 toxic=1
4. 支持合并 clean-negative 短句负例
5. 输出统计报告和清洗后数据集
"""
import argparse
import os
import re
import pandas as pd
from typing import List, Set, Dict
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ========== 关键词定义 ==========

# 地域黑关键词
REGION_DISCRIMINATION_KEYWORDS = [
    '河南人', '东北人', '外地人', '农村人', '乡下人', '土包子',
    '河南骗子', '东北黑', '苏北', '安徽人', '湖北佬', '外省人'
]

# 性别歧视关键词
GENDER_DISCRIMINATION_KEYWORDS = [
    '女司机', '女拳', '田园女权', '婚驴', '仙女', '母狗', '女表',
    '骚货', '破鞋', '绿茶婊', '臭三八', '娘炮', '娘们唧唧'
]

# 种族歧视关键词（仅包含明显贬义词，移除中性描述词）
RACE_DISCRIMINATION_KEYWORDS = [
    '黑鬼', '尼哥', '支那', '小日本', '鬼子', '棒子', '阿三'
]

# 恐同关键词（仅包含贬义词，移除中性描述词）
HOMOPHOBIA_KEYWORDS = [
    '基佬', '死基佬', '搞基', '玻璃', '断背', '娘娘腔', '变态'
]

# 辱骂/攻击关键词
ABUSE_KEYWORDS = [
    '傻逼', '傻B', 'sb', 'SB', '傻X', '煞笔', '沙雕', '智障', '脑残',
    '白痴', '弱智', '废物', '垃圾', '狗东西', '畜生', '贱人', '婊子',
    '操你妈', '草你妈', 'cnm', 'CNM', '妈的', '艹', '靠', '日你',
    '去死', '滚', '滚蛋', '滚开', '死全家', '断子绝孙', '不得好死',
    '王八蛋', '混蛋', '蠢货', '蠢猪', '蠢驴', '猪狗不如'
]

# 威胁关键词
THREAT_KEYWORDS = [
    '杀了你', '弄死你', '揍你', '打死你', '捅你', '砍你', '灭了你',
    '人肉', '寻仇', '报复', '小心点', '走着瞧', '等着瞧', '找你算账'
]

# 色情引流关键词
PORN_DRAIN_KEYWORDS = [
    '约炮', '一夜情', 'yp', 'YP', '上门服务', '技师', '小姐', '嫖娼',
    '援交', '包养', '找小姐', '性服务', '特殊服务', '开房', '炮友',
    '约吗', '有偿', '兼职女', '学生妹'
]

# 广告/诈骗关键词（从 rules.py 扩展）
AD_SCAM_KEYWORDS = [
    '加微信', '加vx', '加VX', '加wx', '加WX', '威信', '扣扣',
    'QQ群', 'qq群', 'QQ：', 'qq：', '刷单', '兼职', '日赚', '月入',
    '小时赚', '躺赚', '低价', '优惠', '返现', '返利', '代购', '批发',
    '折扣', '特价', '代刷', '引流', '推广', '加群', '进群', '扫码',
    '私信我', '私聊', '详情咨询', '了解详情', '手机号', '联系方式'
]

# 合并所有关键词
ALL_KEYWORDS = (
    REGION_DISCRIMINATION_KEYWORDS +
    GENDER_DISCRIMINATION_KEYWORDS +
    RACE_DISCRIMINATION_KEYWORDS +
    HOMOPHOBIA_KEYWORDS +
    ABUSE_KEYWORDS +
    THREAT_KEYWORDS +
    PORN_DRAIN_KEYWORDS +
    AD_SCAM_KEYWORDS
)

# 高置信（强）关键词 - 用于可选的翻标模式
HIGH_CONFIDENCE_KEYWORDS = [
    '傻逼', '傻B', 'sb', 'SB', '操你妈', '草你妈', 'cnm', 'CNM',
    '黑鬼', '尼哥', '死基佬', '婊子', '母狗', '贱人',
    '杀了你', '弄死你', '打死你', '约炮', '一夜情', 'yp', 'YP',
    '刷单', '日赚', '加微信', '加vx', '加VX'
]


# ========== 辅助函数 ==========

def check_harmful_keywords(text: str, keywords: List[str]) -> List[str]:
    """
    检测文本是否包含有害关键词
    
    Args:
        text: 输入文本
        keywords: 关键词列表
        
    Returns:
        List[str]: 命中的关键词列表
    """
    text_lower = text.lower()
    hits = []
    for keyword in keywords:
        if keyword.lower() in text_lower:
            hits.append(keyword)
    return hits


def check_url_pattern(text: str) -> bool:
    """检测 URL 模式"""
    return bool(re.search(r'(https?://|www\.)\S+', text, re.IGNORECASE))


def check_phone_pattern(text: str) -> bool:
    """检测手机号模式"""
    return bool(re.search(r'1[3-9]\d{9}', text))


def check_qq_pattern(text: str) -> bool:
    """检测 QQ 号模式"""
    return bool(re.search(r'(QQ|qq|Qq|扣扣|q群)\s*[:：]?\s*\d{5,11}', text))


def is_harmful_content(text: str, 
                       use_high_confidence_only: bool = False) -> Dict:
    """
    判断文本是否为"明显有害"内容
    
    Args:
        text: 输入文本
        use_high_confidence_only: 是否仅使用高置信关键词
        
    Returns:
        Dict: {
            'is_harmful': bool,
            'keyword_hits': List[str],
            'pattern_hits': List[str],
            'confidence': str  # 'high' or 'medium'
        }
    """
    result = {
        'is_harmful': False,
        'keyword_hits': [],
        'pattern_hits': [],
        'confidence': 'medium'
    }
    
    # 检查关键词
    if use_high_confidence_only:
        keyword_hits = check_harmful_keywords(text, HIGH_CONFIDENCE_KEYWORDS)
    else:
        keyword_hits = check_harmful_keywords(text, ALL_KEYWORDS)
    
    # 检查模式
    pattern_hits = []
    if check_url_pattern(text):
        pattern_hits.append('URL')
    if check_phone_pattern(text):
        pattern_hits.append('Phone')
    if check_qq_pattern(text):
        pattern_hits.append('QQ')
    
    # 判定
    if keyword_hits or pattern_hits:
        result['is_harmful'] = True
        result['keyword_hits'] = keyword_hits
        result['pattern_hits'] = pattern_hits
        
        # 判断置信度
        high_conf_hits = [k for k in keyword_hits if k in HIGH_CONFIDENCE_KEYWORDS]
        if high_conf_hits or len(keyword_hits) >= 2:
            result['confidence'] = 'high'
    
    return result


# ========== 主要功能 ==========

def clean_dataset(
    input_csv: str,
    output_csv: str,
    mode: str = 'drop',
    clean_negatives_csv: str = None,
    clean_neg_ratio: float = 0.1,
    relabel_high_conf: bool = False
) -> Dict:
    """
    清洗数据集
    
    Args:
        input_csv: 输入 CSV 路径
        output_csv: 输出 CSV 路径
        mode: 处理模式 ('drop' 或 'relabel')
        clean_negatives_csv: clean-negative 示例文件路径
        clean_neg_ratio: clean-negative 比例
        relabel_high_conf: 是否对高置信命中翻标
        
    Returns:
        Dict: 统计信息
    """
    print("=" * 80)
    print("数据清洗脚本")
    print("=" * 80)
    print(f"输入文件: {input_csv}")
    print(f"输出文件: {output_csv}")
    print(f"处理模式: {mode}")
    print(f"翻标高置信: {relabel_high_conf}")
    print()
    
    # 1. 加载数据
    print("步骤 1: 加载数据...")
    df = pd.read_csv(input_csv)
    print(f"  总样本数: {len(df)}")
    print(f"  列: {list(df.columns)}")
    print()
    
    # 统计原始数据分布
    if 'source' in df.columns:
        print("数据源分布:")
        print(df['source'].value_counts())
        print()
    
    print("标签分布:")
    print(df['toxic'].value_counts())
    print()
    
    # 2. 筛选 CHSD toxic=0 样本
    print("步骤 2: 筛选 CHSD toxic=0 样本...")
    if 'source' in df.columns:
        chsd_toxic0 = df[(df['source'] == 'CHSD') & (df['toxic'] == 0)].copy()
        print(f"  CHSD toxic=0 样本数: {len(chsd_toxic0)}")
    else:
        # 如果没有 source 列，对所有 toxic=0 进行检测
        print("  警告: 数据中没有 'source' 列，将对所有 toxic=0 样本进行检测")
        chsd_toxic0 = df[df['toxic'] == 0].copy()
        print(f"  toxic=0 样本数: {len(chsd_toxic0)}")
    print()
    
    # 3. 检测有害内容
    print("步骤 3: 检测有害内容...")
    harmful_results = []
    for idx, row in chsd_toxic0.iterrows():
        text = row['content']
        result = is_harmful_content(text, use_high_confidence_only=False)
        if result['is_harmful']:
            harmful_results.append({
                'idx': idx,
                'content': text,
                'keyword_hits': result['keyword_hits'],
                'pattern_hits': result['pattern_hits'],
                'confidence': result['confidence']
            })
    
    print(f"  命中样本数: {len(harmful_results)}")
    if len(chsd_toxic0) > 0:
        print(f"  命中比例: {len(harmful_results) / len(chsd_toxic0) * 100:.2f}%")
    else:
        print(f"  命中比例: N/A (无样本)")
    print()
    
    # 4. 统计高置信和中等置信命中
    high_conf = [r for r in harmful_results if r['confidence'] == 'high']
    medium_conf = [r for r in harmful_results if r['confidence'] == 'medium']
    print(f"  高置信命中: {len(high_conf)}")
    print(f"  中等置信命中: {len(medium_conf)}")
    print()
    
    # 5. 显示示例
    print("命中示例 (前10条):")
    print("-" * 80)
    for i, result in enumerate(harmful_results[:10]):
        print(f"{i+1}. [{result['confidence']}] {result['content'][:60]}...")
        print(f"   关键词: {result['keyword_hits'][:5]}")
        print(f"   模式: {result['pattern_hits']}")
        print()
    
    # 6. 根据模式处理数据
    print("步骤 4: 处理数据...")
    harmful_indices = set([r['idx'] for r in harmful_results])
    
    if mode == 'drop':
        # 剔除模式：移除命中的样本
        df_cleaned = df[~df.index.isin(harmful_indices)].copy()
        print(f"  剔除 {len(harmful_indices)} 个样本")
        removed_count = len(harmful_indices)
        relabeled_count = 0
        
    elif mode == 'relabel':
        # 翻标模式
        df_cleaned = df.copy()
        
        if relabel_high_conf:
            # 仅翻标高置信
            high_conf_indices = set([r['idx'] for r in high_conf])
            df_cleaned.loc[list(high_conf_indices), 'toxic'] = 1
            print(f"  翻标 {len(high_conf_indices)} 个高置信样本为 toxic=1")
            
            # 剔除中等置信
            medium_conf_indices = set([r['idx'] for r in medium_conf])
            df_cleaned = df_cleaned[~df_cleaned.index.isin(medium_conf_indices)]
            print(f"  剔除 {len(medium_conf_indices)} 个中等置信样本")
            
            removed_count = len(medium_conf_indices)
            relabeled_count = len(high_conf_indices)
        else:
            # 全部翻标
            df_cleaned.loc[list(harmful_indices), 'toxic'] = 1
            print(f"  翻标 {len(harmful_indices)} 个样本为 toxic=1")
            removed_count = 0
            relabeled_count = len(harmful_indices)
    else:
        raise ValueError(f"不支持的模式: {mode}")
    
    print()
    
    # 7. 加入 clean-negative 样本（可选）
    clean_neg_added = 0
    if clean_negatives_csv and os.path.exists(clean_negatives_csv):
        print("步骤 5: 加入 clean-negative 样本...")
        clean_neg_df = pd.read_csv(clean_negatives_csv)
        
        # 确保列名一致
        if 'content' not in clean_neg_df.columns:
            print("  警告: clean-negative 文件缺少 'content' 列，跳过")
        else:
            if 'toxic' not in clean_neg_df.columns:
                clean_neg_df['toxic'] = 0
            if 'source' not in clean_neg_df.columns and 'source' in df_cleaned.columns:
                clean_neg_df['source'] = 'clean_neg'
            
            # 按比例采样
            n_samples = int(len(df_cleaned) * clean_neg_ratio)
            if n_samples > len(clean_neg_df):
                n_samples = len(clean_neg_df)
            
            clean_neg_sample = clean_neg_df.sample(n=n_samples, random_state=42)
            df_cleaned = pd.concat([df_cleaned, clean_neg_sample], ignore_index=True)
            clean_neg_added = len(clean_neg_sample)
            
            print(f"  加入 {clean_neg_added} 个 clean-negative 样本")
            print()
    
    # 8. 保存清洗后的数据
    print("步骤 6: 保存清洗后数据...")
    df_cleaned.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"  已保存到: {output_csv}")
    print()
    
    # 9. 统计信息
    stats = {
        'input_samples': len(df),
        'chsd_toxic0_samples': len(chsd_toxic0),
        'harmful_detected': len(harmful_results),
        'high_confidence': len(high_conf),
        'medium_confidence': len(medium_conf),
        'removed_samples': removed_count,
        'relabeled_samples': relabeled_count,
        'clean_neg_added': clean_neg_added,
        'output_samples': len(df_cleaned),
        'output_toxic_count': df_cleaned['toxic'].sum(),
        'output_toxic_ratio': df_cleaned['toxic'].mean()
    }
    
    print("=" * 80)
    print("清洗完成 - 统计报告")
    print("=" * 80)
    print(f"输入样本数:         {stats['input_samples']}")
    print(f"CHSD toxic=0 样本:  {stats['chsd_toxic0_samples']}")
    
    # 避免除以零
    if stats['chsd_toxic0_samples'] > 0:
        harmful_pct = stats['harmful_detected'] / stats['chsd_toxic0_samples'] * 100
        print(f"检测到有害:         {stats['harmful_detected']} ({harmful_pct:.2f}%)")
    else:
        print(f"检测到有害:         {stats['harmful_detected']} (N/A)")
    
    print(f"  - 高置信:         {stats['high_confidence']}")
    print(f"  - 中等置信:       {stats['medium_confidence']}")
    print(f"剔除样本数:         {stats['removed_samples']}")
    print(f"翻标样本数:         {stats['relabeled_samples']}")
    print(f"加入 clean-neg:     {stats['clean_neg_added']}")
    print(f"输出样本数:         {stats['output_samples']}")
    print(f"输出有毒样本:       {stats['output_toxic_count']} ({stats['output_toxic_ratio']*100:.2f}%)")
    print()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='数据清洗与构建脚本 - 改进广义有害内容检测训练数据'
    )
    
    # 输入输出参数
    parser.add_argument(
        '--input', type=str,
        default='data/mixture_toxicn_chsd.csv',
        help='输入 CSV 文件路径 (默认: data/mixture_toxicn_chsd.csv)'
    )
    parser.add_argument(
        '--output', type=str,
        default='data/mixture_cleaned.csv',
        help='输出 CSV 文件路径 (默认: data/mixture_cleaned.csv)'
    )
    
    # 处理模式
    parser.add_argument(
        '--mode', type=str, choices=['drop', 'relabel'],
        default='drop',
        help='处理模式: drop=剔除命中样本(默认), relabel=翻标为 toxic=1'
    )
    parser.add_argument(
        '--relabel-high-conf', action='store_true',
        help='在 relabel 模式下，仅翻标高置信样本，剔除中等置信'
    )
    
    # clean-negative 参数
    parser.add_argument(
        '--clean-negatives', type=str,
        default='data/clean_negatives.csv',
        help='clean-negative 示例文件路径 (默认: data/clean_negatives.csv)'
    )
    parser.add_argument(
        '--clean-neg-ratio', type=float, default=0.0,
        help='加入 clean-negative 的比例 (默认: 0.0, 不加入)'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1
    
    # 执行清洗
    stats = clean_dataset(
        input_csv=args.input,
        output_csv=args.output,
        mode=args.mode,
        clean_negatives_csv=args.clean_negatives if args.clean_neg_ratio > 0 else None,
        clean_neg_ratio=args.clean_neg_ratio,
        relabel_high_conf=args.relabel_high_conf
    )
    
    print("=" * 80)
    print("全部完成!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())
