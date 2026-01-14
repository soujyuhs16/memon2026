"""
推理预测模块
Inference and prediction utilities
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union
import os

from .rules import check_rules, merge_predictions


class ToxicClassifier:
    """中文有毒评论分类器"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        初始化分类器
        
        Args:
            model_path: 模型路径
            device: 设备 (cuda/cpu)，默认自动检测
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
    
    def predict_single(self, 
                       text: str, 
                       threshold: float = 0.5,
                       use_rules: bool = True,
                       max_length: int = 128) -> Dict:
        """
        单条文本预测
        
        Args:
            text: 输入文本
            threshold: 判定阈值
            use_rules: 是否使用规则融合
            max_length: 最大序列长度
            
        Returns:
            dict: {
                'text': str,
                'model_prob': float,
                'rule_hits': list,
                'rule_score': float,
                'final_prob': float,
                'pred': int,
                'threshold': float
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
            # 融合预测
            final_prob = merge_predictions(prob, rule_score)
        else:
            rule_hits = []
            rule_score = 0.0
            final_prob = prob
        
        # 判定
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
                      max_length: int = 128,
                      batch_size: int = 32) -> List[Dict]:
        """
        批量文本预测
        
        Args:
            texts: 文本列表
            threshold: 判定阈值
            use_rules: 是否使用规则融合
            max_length: 最大序列长度
            batch_size: 批次大小
            
        Returns:
            List[Dict]: 预测结果列表
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
                    final_prob = merge_predictions(prob, rule_score)
                else:
                    rule_hits = []
                    rule_score = 0.0
                    final_prob = prob
                
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


def load_model(model_path: str, device: str = None) -> ToxicClassifier:
    """
    便捷函数：加载模型
    
    Args:
        model_path: 模型路径
        device: 设备
        
    Returns:
        ToxicClassifier: 分类器实例
    """
    return ToxicClassifier(model_path, device)


if __name__ == '__main__':
    # 测试用例
    print("预测模块测试")
    print("=" * 60)
    print("\n注意: 需要先训练模型才能运行此测试")
    print("示例用法:")
    print("""
    from src.predict import load_model
    
    # 加载模型
    classifier = load_model('outputs/model')
    
    # 单条预测
    result = classifier.predict_single('这个产品很好用')
    print(result)
    
    # 批量预测
    texts = ['文本1', '文本2', '文本3']
    results = classifier.predict_batch(texts)
    for r in results:
        print(r)
    """)
