"""
Streamlit 管理界面
Web UI for toxic comment classification
"""
import streamlit as st
import pandas as pd
import os
import sys

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.predict import load_predictor


# 配置
MODEL_PATH = os.environ.get('MODEL_PATH', 'outputs/model')


# 页面配置
st.set_page_config(
    page_title="中文评论审核系统",
    page_icon="🛡️",
    layout="wide"
)


# 样式
st.markdown("""
<style>
    .stAlert {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 10px;
    }
    .toxic-badge {
        background-color: #dc3545;
        color: white;
        padding: 5px 10px;
        border-radius: 3px;
        font-weight: bold;
    }
    .safe-badge {
        background-color: #28a745;
        color: white;
        padding: 5px 10px;
        border-radius: 3px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    """加载模型（缓存）"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"模型未找到: {MODEL_PATH}")
        st.info("请先运行训练脚本: `python src/train.py`")
        st.stop()
    
    with st.spinner('加载模型中...'):
        return load_predictor(MODEL_PATH)


def main():
    # 标题和警告
    st.title("🛡️ 中文评论审核系统")
    st.markdown("---")
    
    # 警告信息
    st.warning(
        "⚠️ **重要提示**: 本系统用于检测有害内容（辱骂/仇恨/引流广告），数据包含敏感内容，"
        "仅供科研和学术用途使用。请勿用于商业目的。"
    )
    
    # 加载模型
    try:
        classifier = load_classifier()
        st.success("✅ 模型加载成功")
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        st.stop()
    
    # 侧边栏配置
    st.sidebar.header("⚙️ 配置")
    
    # 显示默认阈值（来自模型校准）
    default_threshold = classifier.default_threshold
    st.sidebar.info(f"📊 模型校准阈值: {default_threshold:.3f}")
    
    threshold = st.sidebar.slider(
        "判定阈值",
        min_value=0.0,
        max_value=1.0,
        value=default_threshold,
        step=0.05,
        help="概率高于此阈值将被判定为有害内容（默认使用模型校准阈值）"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**模型信息**\n\n"
        f"- 模型路径: `{MODEL_PATH}`\n"
        f"- 基线模型: hfl/chinese-roberta-wwm-ext\n"
        f"- 任务: 广义有害内容二分类\n"
        f"- 规则融合: 已启用\n"
        f"- 分类: 辱骂/仇恨/引流广告"
    )
    
    # 主界面选项卡
    tab1, tab2 = st.tabs(["📝 单条预测", "📊 批量预测"])
    
    # ===== Tab 1: 单条预测 =====
    with tab1:
        st.header("单条文本预测")
        
        # 输入框
        text_input = st.text_area(
            "输入评论文本",
            height=150,
            placeholder="例如：这个产品很好用...",
            help="输入需要检测的评论文本"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            predict_button = st.button("🔍 预测", type="primary", use_container_width=True)
        
        if predict_button and text_input.strip():
            with st.spinner('预测中...'):
                result = classifier.predict_one(
                    text_input,
                    threshold=threshold,
                    use_rules=True
                )
            
            st.markdown("---")
            st.subheader("预测结果")
            
            # 结果展示
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "模型概率",
                    f"{result['model_prob']:.3f}"
                )
            
            with col2:
                st.metric(
                    "规则分数",
                    f"{result['rule_score']:.3f}"
                )
            
            with col3:
                st.metric(
                    "最终概率",
                    f"{result['final_prob']:.3f}"
                )
            
            # 判定结果
            if result['pred'] == 1:
                st.markdown(
                    '<div class="toxic-badge">🚫 有害内容（辱骂/仇恨/引流广告）</div>',
                    unsafe_allow_html=True
                )
                # 显示类别提示
                if result.get('category_hint'):
                    st.info(f"**可能类别**: {result['category_hint']}")
            else:
                st.markdown(
                    '<div class="safe-badge">✅ 安全内容</div>',
                    unsafe_allow_html=True
                )
            
            # 规则命中
            if result['rule_hits']:
                st.markdown("**规则命中:**")
                for hit in result['rule_hits']:
                    st.markdown(f"- `{hit}`")
            else:
                st.info("未命中任何规则")
            
            # 详细信息
            with st.expander("📋 详细信息"):
                st.json(result)
        
        elif predict_button:
            st.warning("请输入文本后再预测")
    
    # ===== Tab 2: 批量预测 =====
    with tab2:
        st.header("批量CSV文件预测")
        
        st.markdown("""
        **使用说明:**
        1. 上传包含评论文本的 CSV 文件
        2. 选择文本列名（默认为 'content'）
        3. 点击预测按钮
        4. 下载预测结果
        """)
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "选择CSV文件",
            type=['csv'],
            help="文件必须包含至少一列文本数据"
        )
        
        if uploaded_file is not None:
            # 读取CSV
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ 文件加载成功: {len(df)} 行")
                
                # 预览数据
                with st.expander("📋 数据预览（前10行）"):
                    st.dataframe(df.head(10))
                
                # 选择文本列
                text_column = st.selectbox(
                    "选择文本列",
                    options=df.columns.tolist(),
                    index=df.columns.tolist().index('content') if 'content' in df.columns else 0
                )
                
                # 预测按钮
                if st.button("🔍 批量预测", type="primary"):
                    # 获取文本
                    texts = df[text_column].fillna('').astype(str).tolist()
                    
                    # 批量预测
                    with st.spinner(f'预测中... (共 {len(texts)} 条)'):
                        results = classifier.predict_batch(
                            texts,
                            threshold=threshold,
                            use_rules=True
                        )
                    
                    # 构建结果DataFrame
                    result_df = pd.DataFrame(results)
                    
                    # 合并原始数据
                    output_df = df.copy()
                    output_df['model_prob'] = result_df['model_prob']
                    output_df['rule_hits'] = result_df['rule_hits'].apply(
                        lambda x: ','.join(x) if x else ''
                    )
                    output_df['rule_score'] = result_df['rule_score']
                    output_df['final_prob'] = result_df['final_prob']
                    output_df['pred'] = result_df['pred']
                    output_df['category_hint'] = result_df['category_hint']
                    
                    st.success("✅ 预测完成!")
                    
                    # 统计信息
                    st.markdown("---")
                    st.subheader("📊 统计信息")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("总样本数", len(output_df))
                    
                    with col2:
                        harmful_count = (output_df['pred'] == 1).sum()
                        st.metric("有害内容", harmful_count)
                    
                    with col3:
                        safe_count = (output_df['pred'] == 0).sum()
                        st.metric("安全内容", safe_count)
                    
                    # 预览结果
                    st.markdown("---")
                    st.subheader("预测结果预览")
                    st.dataframe(output_df.head(20))
                    
                    # 下载按钮
                    csv_buffer = output_df.to_csv(index=False, encoding='utf-8')
                    st.download_button(
                        label="📥 下载完整预测结果",
                        data=csv_buffer,
                        file_name=f"predictions_{uploaded_file.name}",
                        mime="text/csv",
                        type="primary"
                    )
            
            except Exception as e:
                st.error(f"❌ 处理文件时出错: {str(e)}")
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        <p>
            <strong>数据集:</strong> ToxiCN (CC BY-NC-ND 4.0) | 
            <strong>引用:</strong> ACL 2023 | 
            <strong>用途:</strong> 仅限科研非商用
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
