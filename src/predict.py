"""
推理预测模块
Inference and prediction utilities
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union, Optional
import os

from .rules import check_rules, merge_predictions


class Predictor:
    """
    中文有毒评论分类器
    Chinese toxic comment classifier
    
    Usage:
        >>> predictor = Predictor('outputs/model')
        >>> result = predictor.predict_one('加vx领资料，低价代刷')
        >>> print(result['pred'])
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        初始化分类器
        Initialize the classifier
        
        Args:
            model_path: 模型路径 (Path to the model directory)
            device: 设备 (cuda/cpu)，默认自动检测 (Device, auto-detected if None)
        """
        self.model_path = model_path
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"加载模型从: {model_path}")
        print(f"使用设备: {self.device}")
        
        # 加载 tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("模型加载完成")
    
    def predict_one(self, 
                    text: str, 
                    threshold: float = 0.5,
                    use_rules: bool = True,
                    rule_override: bool = False,
                    max_length: int = 128) -> Dict:
        """
        单条文本预测
        Single text prediction
        
        Args:
            text: 输入文本 (Input text)
            threshold: 判定阈值 (Classification threshold, default 0.5)
            use_rules: 是否使用规则融合 (Whether to use rule fusion, default True)
            rule_override: 规则命中时是否直接判定为有毒 (Whether rule hits override model, default False)
            max_length: 最大序列长度 (Max sequence length, default 128)
            
        Returns:
            dict: {
                'text': str,           # 原文本
                'model_prob': float,   # 模型预测概率 (0~1)
                'rule_hits': list,     # 命中的规则列表
                'rule_score': float,   # 规则得分 (0~1)
                'final_prob': float,   # 融合后最终概率
                'pred': int,           # 预测标签 (0=正常, 1=有毒)
                'threshold': float     # 使用的阈值
            }
        """
        # 模型预测
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # 二分类 sigmoid
            prob = torch.sigmoid(logits).squeeze().item()
        
        # 规则检测
        if use_rules:
            rule_result = check_rules(text)
            rule_hits = rule_result['hits']
            rule_score = rule_result['score']
            # 融合预测（通过 merge_predictions 应用 rule_override 逻辑）
            final_prob = merge_predictions(prob, rule_score, rule_override=rule_override)
        else:
            rule_hits = []
            rule_score = 0.0
            final_prob = prob
        
        # 判定 (Classification)
        pred = 1 if final_prob >= threshold else 0
        
        return {
            'text': text,
            'model_prob': prob,
            'rule_hits': rule_hits,
            'rule_score': rule_score,
            'final_prob': final_prob,
            'pred': pred,
            'threshold': threshold
        }
    
    def predict_batch(self, 
                      texts: List[str], 
                      threshold: float = 0.5,
                      use_rules: bool = True,
                      rule_override: bool = False,
                      max_length: int = 128,
                      batch_size: int = 32) -> List[Dict]:
        """
        批量文本预测
        Batch text prediction
        
        Args:
            texts: 文本列表 (List of texts)
            threshold: 判定阈值 (Classification threshold, default 0.5)
            use_rules: 是否使用规则融合 (Whether to use rule fusion, default True)
            rule_override: 规则命中时是否直接判定为有毒 (Whether rule hits override model, default False)
            max_length: 最大序列长度 (Max sequence length, default 128)
            batch_size: 批次大小 (Batch size for processing, default 32)
            
        Returns:
            List[Dict]: 预测结果列表 (List of prediction results)
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 批量tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 批量预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # 确保是数组
            if len(batch_texts) == 1:
                probs = [probs.item()]
            else:
                probs = probs.tolist()
            
            # 逐条处理规则和融合
            for text, prob in zip(batch_texts, probs):
                if use_rules:
                    rule_result = check_rules(text)
                    rule_hits = rule_result['hits']
                    rule_score = rule_result['score']
                    # 融合预测（通过 merge_predictions 应用 rule_override 逻辑）
                    final_prob = merge_predictions(prob, rule_score, rule_override=rule_override)
                else:
                    rule_hits = []
                    rule_score = 0.0
                    final_prob = prob
                
                # 判定 (Classification)
                pred = 1 if final_prob >= threshold else 0
                
                results.append({
                    'text': text,
                    'model_prob': prob,
                    'rule_hits': rule_hits,
                    'rule_score': rule_score,
                    'final_prob': final_prob,
                    'pred': pred,
                    'threshold': threshold
                })
        
        return results
    
    # Backward compatibility alias
    def predict_single(self, *args, **kwargs) -> Dict:
        """向后兼容别名 (Backward compatibility alias for predict_one)"""
        return self.predict_one(*args, **kwargs)


def load_predictor(model_dir: str, device: Optional[str] = None) -> Predictor:
    """
    加载预测器（推荐接口）
    Load predictor (recommended interface)
    
    Args:
        model_dir: 模型目录路径 (Path to model directory, e.g., 'outputs/model')
        device: 设备 (Device: 'cuda', 'cpu', or None for auto-detection)
        
    Returns:
        Predictor: 预测器实例 (Predictor instance)
    
    Example:
        >>> predictor = load_predictor('outputs/model')
        >>> result = predictor.predict_one('加vx领资料，低价代刷', threshold=0.5)
        >>> print(result)
    """
    return Predictor(model_dir, device)


def load_model(model_path: str, device: Optional[str] = None) -> Predictor:
    """
    便捷函数：加载模型（向后兼容别名）
    Convenience function: Load model (backward compatibility alias)
    
    Args:
        model_path: 模型路径 (Model path)
        device: 设备 (Device)
        
    Returns:
        Predictor: 分类器实例 (Predictor instance)
    """
    return Predictor(model_path, device)


# Backward compatibility aliases
ToxicClassifier = Predictor  # 向后兼容旧类名


if __name__ == '__main__':
    # 测试用例
    print("预测模块测试")
    print("=" * 60)
    print("\n注意: 需要先训练模型才能运行此测试")
    print("示例用法:")
    print("""
    from src.predict import load_predictor
    
    # 加载模型
    predictor = load_predictor('outputs/model')
    
    # 单条预测
    result = predictor.predict_one('这个产品很好用', threshold=0.5)
    print(result)
    
    # 批量预测
    texts = ['文本1', '文本2', '文本3']
    results = predictor.predict_batch(texts, threshold=0.5)
    for r in results:
        print(r)
    """)
