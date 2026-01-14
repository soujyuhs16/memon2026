"""
FastAPI 推理服务
REST API service for toxic comment classification
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import io
import os
import sys

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.predict import load_model


# 配置
MODEL_PATH = os.environ.get('MODEL_PATH', 'outputs/model')
DEFAULT_THRESHOLD = 0.5

# 全局模型实例（延迟加载）
classifier = None


def get_classifier():
    """获取或初始化分类器"""
    global classifier
    if classifier is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"模型未找到: {MODEL_PATH}\n"
                "请先运行训练脚本: python src/train.py"
            )
        classifier = load_model(MODEL_PATH)
    return classifier


# FastAPI app
app = FastAPI(
    title="中文评论审核 API",
    description="基于 Transformer 的中文有毒评论分类服务（ToxiCN）",
    version="1.0.0"
)


# 请求/响应模型
class PredictRequest(BaseModel):
    text: str
    threshold: Optional[float] = DEFAULT_THRESHOLD
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "这个产品很好用",
                "threshold": 0.5
            }
        }


class PredictResponse(BaseModel):
    text: str
    model_prob: float
    rule_hits: List[str]
    rule_score: float
    final_prob: float
    pred: int
    threshold: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "这个产品很好用",
                "model_prob": 0.05,
                "rule_hits": [],
                "rule_score": 0.0,
                "final_prob": 0.05,
                "pred": 0,
                "threshold": 0.5
            }
        }


@app.get("/")
def root():
    """API 根路径"""
    return {
        "message": "中文评论审核 API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict - 单条文本预测",
            "batch_predict": "POST /batch_predict - 批量CSV预测"
        },
        "model_path": MODEL_PATH,
        "warning": "本API包含敏感内容，仅用于科研目的"
    }


@app.get("/health")
def health_check():
    """健康检查"""
    try:
        get_classifier()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "model_loaded": False}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    单条文本预测
    
    - **text**: 输入文本
    - **threshold**: 判定阈值 (可选，默认0.5)
    
    返回模型概率、规则命中、最终概率和预测标签
    """
    try:
        clf = get_classifier()
        result = clf.predict_single(
            request.text,
            threshold=request.threshold,
            use_rules=True
        )
        return PredictResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(
    file: UploadFile = File(...),
    threshold: float = DEFAULT_THRESHOLD,
    text_column: str = 'content'
):
    """
    批量CSV文件预测
    
    - **file**: CSV文件（必须包含文本列）
    - **threshold**: 判定阈值 (可选，默认0.5)
    - **text_column**: 文本列名 (可选，默认'content')
    
    返回带预测结果的CSV文件
    """
    try:
        # 读取上传的CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 检查文本列是否存在
        if text_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"CSV文件中未找到列 '{text_column}'。可用列: {list(df.columns)}"
            )
        
        # 获取文本列表
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # 批量预测
        clf = get_classifier()
        results = clf.predict_batch(texts, threshold=threshold, use_rules=True)
        
        # 构建结果DataFrame
        result_df = pd.DataFrame(results)
        
        # 合并原始数据和预测结果
        # 保留原始所有列，添加预测列
        output_df = df.copy()
        output_df['model_prob'] = result_df['model_prob']
        output_df['rule_hits'] = result_df['rule_hits'].apply(
            lambda x: ','.join(x) if x else ''
        )
        output_df['rule_score'] = result_df['rule_score']
        output_df['final_prob'] = result_df['final_prob']
        output_df['pred'] = result_df['pred']
        
        # 转换为CSV
        output_buffer = io.StringIO()
        output_df.to_csv(output_buffer, index=False, encoding='utf-8')
        output_buffer.seek(0)
        
        # 返回文件下载
        return StreamingResponse(
            io.BytesIO(output_buffer.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=predictions_{file.filename}"
            }
        )
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV文件为空")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="CSV文件格式错误")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")


if __name__ == '__main__':
    import uvicorn
    print("=" * 80)
    print("启动 FastAPI 服务")
    print("=" * 80)
    print(f"模型路径: {MODEL_PATH}")
    print(f"默认阈值: {DEFAULT_THRESHOLD}")
    print()
    print("访问 http://localhost:8000/docs 查看 API 文档")
    print("=" * 80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
