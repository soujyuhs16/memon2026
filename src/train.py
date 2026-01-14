"""
训练脚本
Training script for toxic comment classification
"""
import argparse
import os
import json
import inspect
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, Value

from data import load_toxicn_data, split_data
from rules import check_rules, merge_predictions


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    # sigmoid 转换
    probs = 1 / (1 + np.exp(-predictions))
    preds = (probs > 0.5).astype(int).flatten()
    labels = labels.flatten()
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def tokenize_function(examples, tokenizer, max_length):
    """Tokenization function"""
    return tokenizer(
        examples['text'], 
        truncation=True, 
        max_length=max_length,
        padding=False  # 使用 DataCollator 动态padding
    )


def main():
    parser = argparse.ArgumentParser(description='训练中文有毒评论分类模型')
    
    # 数据参数
    parser.add_argument('--csv_path', type=str, default='data/ToxiCN_1.0.csv',
                        help='ToxiCN 数据集路径')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--dev_size', type=float, default=0.1,
                        help='验证集比例（从训练集中切分）')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, 
                        default='hfl/chinese-roberta-wwm-ext',
                        help='预训练模型名称')
    parser.add_argument('--max_length', type=int, default=128,
                        help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='训练批次大小')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='评估批次大小')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='预测阈值')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("中文有毒评论分类模型训练")
    print("=" * 80)
    print(f"\n配置参数:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    # 1. 加载并划分数据
    print("=" * 80)
    print("步骤 1: 加载数据")
    print("=" * 80)
    df = load_toxicn_data(args.csv_path)
    print(f"总样本数: {len(df)}")
    print(f"有毒样本: {df['toxic'].sum()} ({df['toxic'].mean()*100:.2f}%)")
    print()
    
    train_df, dev_df, test_df = split_data(
        df, 
        test_size=args.test_size,
        dev_size=args.dev_size,
        random_state=args.seed
    )
    print()
    
    # 2. 转换为 Datasets 格式
    print("=" * 80)
    print("步骤 2: 准备数据集")
    print("=" * 80)
    
    train_dataset = Dataset.from_dict({
        'text': train_df['content'].tolist(),
        'label': train_df['toxic'].tolist()
    })
    dev_dataset = Dataset.from_dict({
        'text': dev_df['content'].tolist(),
        'label': dev_df['toxic'].tolist()
    })
    test_dataset = Dataset.from_dict({
        'text': test_df['content'].tolist(),
        'label': test_df['toxic'].tolist()
    })
    
    # 重命名列: label -> labels (Trainer 期望的列名)
    train_dataset = train_dataset.rename_column('label', 'labels')
    dev_dataset = dev_dataset.rename_column('label', 'labels')
    test_dataset = test_dataset.rename_column('label', 'labels')
    
    # 转换 labels 为 float32 (BCEWithLogitsLoss 要求)
    train_dataset = train_dataset.cast_column('labels', Value('float32'))
    dev_dataset = dev_dataset.cast_column('labels', Value('float32'))
    test_dataset = test_dataset.cast_column('labels', Value('float32'))
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(dev_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    print()
    
    # 3. 加载 tokenizer 和模型
    print("=" * 80)
    print("步骤 3: 加载模型")
    print("=" * 80)
    print(f"加载预训练模型: {args.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1  # 二分类单输出
    )
    
    print("模型加载完成")
    print()
    
    # 4. Tokenize 数据
    print("=" * 80)
    print("步骤 4: Tokenize 数据")
    print("=" * 80)
    
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=['text']
    )
    tokenized_dev = dev_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=['text']
    )
    tokenized_test = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=['text']
    )
    
    print("Tokenization 完成")
    
    # 数据类型检查和日志
    print("\n[Sanity Check] 标签数据类型检查:")
    print(f"  训练集 labels dtype: {tokenized_train.features['labels'].dtype}")
    print(f"  验证集 labels dtype: {tokenized_dev.features['labels'].dtype}")
    print(f"  测试集 labels dtype: {tokenized_test.features['labels'].dtype}")
    if len(tokenized_train) > 0:
        print(f"  训练集前3个标签示例: {tokenized_train['labels'][:3]}")
    print()
    
    # 5. 设置训练参数
    print("=" * 80)
    print("步骤 5: 配置训练")
    print("=" * 80)
    
    model_output_dir = os.path.join(args.output_dir, 'model')
    
    # 兼容不同版本的 transformers 参数名称
    # Compatible with different transformers versions for evaluation strategy parameter
    # 旧版本使用 'evaluation_strategy'，新版本使用 'eval_strategy'
    # Older versions use 'evaluation_strategy', newer versions use 'eval_strategy'
    training_args_kwargs = {
        'output_dir': model_output_dir,
        'num_train_epochs': args.epochs,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.eval_batch_size,
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'save_strategy': "epoch",
        'load_best_model_at_end': True,
        'metric_for_best_model': "f1",
        'seed': args.seed,
        'logging_dir': os.path.join(args.output_dir, 'logs'),
        'logging_steps': 50,
        'save_total_limit': 2,
        'report_to': "none"  # 不使用wandb等
    }
    
    # 检查 TrainingArguments 支持哪个参数名
    # Check which parameter name is supported by TrainingArguments
    sig = inspect.signature(TrainingArguments.__init__)
    if 'evaluation_strategy' in sig.parameters:
        # 旧版本 transformers
        # Older transformers versions
        training_args_kwargs['evaluation_strategy'] = "epoch"
    elif 'eval_strategy' in sig.parameters:
        # 新版本 transformers
        # Newer transformers versions
        training_args_kwargs['eval_strategy'] = "epoch"
    else:
        # 如果两个参数都不支持，报错提示用户
        # If neither parameter is supported, raise an informative error
        raise ValueError(
            "TrainingArguments does not support 'evaluation_strategy' or 'eval_strategy' parameter. "
            "Please upgrade your transformers library: pip install --upgrade transformers"
        )
    
    training_args = TrainingArguments(**training_args_kwargs)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("训练器初始化完成")
    print()
    
    # 6. 开始训练
    print("=" * 80)
    print("步骤 6: 开始训练")
    print("=" * 80)
    
    trainer.train()
    
    print("\n训练完成!")
    print()
    
    # 7. 评估
    print("=" * 80)
    print("步骤 7: 评估模型")
    print("=" * 80)
    
    # 验证集评估
    print("\n验证集评估:")
    dev_metrics = trainer.evaluate(tokenized_dev)
    print(json.dumps(dev_metrics, indent=2))
    
    # 保存验证集指标
    dev_metrics_path = os.path.join(args.output_dir, 'metrics_dev.json')
    with open(dev_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(dev_metrics, f, indent=2, ensure_ascii=False)
    print(f"验证集指标已保存: {dev_metrics_path}")
    
    # 测试集评估
    print("\n测试集评估:")
    test_metrics = trainer.evaluate(tokenized_test)
    print(json.dumps(test_metrics, indent=2))
    
    # 保存测试集指标
    test_metrics_path = os.path.join(args.output_dir, 'metrics_test.json')
    with open(test_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print(f"测试集指标已保存: {test_metrics_path}")
    print()
    
    # 8. 保存模型
    print("=" * 80)
    print("步骤 8: 保存模型")
    print("=" * 80)
    
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"模型已保存到: {model_output_dir}")
    print()
    
    # 9. 生成测试集预测
    print("=" * 80)
    print("步骤 9: 生成测试集预测")
    print("=" * 80)
    
    # 获取模型预测
    predictions = trainer.predict(tokenized_test)
    logits = predictions.predictions
    probs = 1 / (1 + np.exp(-logits.flatten()))
    
    # 准备预测结果 DataFrame
    test_predictions = test_df.copy()
    test_predictions['model_prob'] = probs
    
    # 添加规则检测
    print("应用规则检测...")
    rule_results = test_predictions['content'].apply(check_rules)
    test_predictions['rule_hits'] = rule_results.apply(lambda x: ','.join(x['hits']) if x['hits'] else '')
    test_predictions['rule_score'] = rule_results.apply(lambda x: x['score'])
    
    # 融合预测
    test_predictions['final_prob'] = test_predictions.apply(
        lambda row: merge_predictions(row['model_prob'], row['rule_score']),
        axis=1
    )
    
    # 生成预测标签
    test_predictions['pred'] = (test_predictions['final_prob'] >= args.threshold).astype(int)
    
    # 重命名列
    test_predictions = test_predictions.rename(columns={'toxic': 'label'})
    
    # 选择输出列
    output_columns = ['content', 'label', 'model_prob', 'rule_score', 
                      'final_prob', 'pred', 'rule_hits']
    test_predictions = test_predictions[output_columns]
    
    # 保存预测结果
    predictions_path = os.path.join(args.output_dir, 'test_predictions.csv')
    test_predictions.to_csv(predictions_path, index=False, encoding='utf-8')
    print(f"测试集预测已保存: {predictions_path}")
    
    # 打印样例
    print("\n预测样例（前5条）:")
    print(test_predictions.head())
    print()
    
    # 最终统计
    print("=" * 80)
    print("训练完成总结")
    print("=" * 80)
    print(f"模型保存路径: {model_output_dir}")
    print(f"验证集指标: {dev_metrics_path}")
    print(f"测试集指标: {test_metrics_path}")
    print(f"测试集预测: {predictions_path}")
    print()
    print(f"测试集最终性能:")
    print(f"  准确率: {test_metrics.get('eval_accuracy', 0):.4f}")
    print(f"  精确率: {test_metrics.get('eval_precision', 0):.4f}")
    print(f"  召回率: {test_metrics.get('eval_recall', 0):.4f}")
    print(f"  F1分数: {test_metrics.get('eval_f1', 0):.4f}")
    print(f"  AUC: {test_metrics.get('eval_auc', 0):.4f}")
    print()
    print("=" * 80)
    print("全部完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
