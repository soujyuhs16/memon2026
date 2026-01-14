#!/bin/bash
# 系统集成测试脚本
# Integration test script for the Chinese comment moderation system

set -e  # Exit on error

echo "=================================="
echo "中文评论审核系统 - 集成测试"
echo "=================================="
echo ""

# 检查环境
echo "步骤 1: 检查环境..."
python --version
echo ""

# 检查项目结构
echo "步骤 2: 检查项目结构..."
echo "项目文件结构:"
ls -R | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\//--/g' -e 's/^/   /' -e 's/-/|/'
echo ""

# 测试规则模块
echo "步骤 3: 测试规则模块..."
python src/rules.py
echo "✅ 规则模块测试通过"
echo ""

# 检查语法
echo "步骤 4: 检查所有模块语法..."
python -m py_compile src/data.py
echo "✅ data.py 语法正确"
python -m py_compile src/predict.py
echo "✅ predict.py 语法正确"
python -m py_compile src/train.py
echo "✅ train.py 语法正确"
python -m py_compile api/main.py
echo "✅ api/main.py 语法正确"
python -m py_compile app/app.py
echo "✅ app/app.py 语法正确"
echo ""

# 显示训练脚本帮助
echo "步骤 5: 显示训练脚本参数..."
echo "注意: 需要安装依赖才能运行完整测试"
echo "使用 'pip install -r requirements.txt' 安装依赖"
echo ""

echo "=================================="
echo "基础测试完成!"
echo "=================================="
echo ""
echo "后续步骤（需要先安装依赖和数据）:"
echo "1. 安装依赖: pip install -r requirements.txt"
echo "2. 准备数据: 将 ToxiCN_1.0.csv 放入 data/ 目录"
echo "3. 训练模型: python src/train.py"
echo "4. 启动API: uvicorn api.main:app --reload"
echo "5. 启动UI: streamlit run app/app.py"
echo ""
