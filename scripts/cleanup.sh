#!/bin/bash
# 清理脚本

echo "清理轨形染色系统临时文件"
echo "========================"

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

# 询问确认
read -p "确定要清理临时文件吗？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消清理"
    exit 0
fi

# 清理目录
DIRS_TO_CLEAN=(
    "$PROJECT_DIR/__pycache__"
    "$PROJECT_DIR/src/__pycache__"
    "$PROJECT_DIR/src/core/__pycache__"
    "$PROJECT_DIR/src/pipeline/__pycache__"
    "$PROJECT_DIR/src/utils/__pycache__"
    "$PROJECT_DIR/src/visualization/__pycache__"
    "$PROJECT_DIR/examples/__pycache__"
    "$PROJECT_DIR/tests/__pycache__"
    "$PROJECT_DIR/.pytest_cache"
    "$PROJECT_DIR/build"
    "$PROJECT_DIR/dist"
    "$PROJECT_DIR/*.egg-info"
    "$PROJECT_DIR/.coverage"
    "$PROJECT_DIR/htmlcov"
    "$PROJECT_DIR/.mypy_cache"
)

echo "清理缓存目录..."
for dir in "${DIRS_TO_CLEAN[@]}"; do
    if [ -e "$dir" ]; then
        echo "清理: $dir"
        rm -rf "$dir"
    fi
done

# 清理日志文件（可选）
read -p "是否清理日志文件？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    LOG_DIR="$PROJECT_DIR/logs"
    if [ -d "$LOG_DIR" ]; then
        echo "清理日志目录..."
        rm -rf "$LOG_DIR"/*
    fi
fi

# 清理临时结果（可选）
read -p "是否清理临时结果文件？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    TEMP_RESULTS="$PROJECT_DIR/results/temp_*"
    if ls $TEMP_RESULTS 1> /dev/null 2>&1; then
        echo "清理临时结果..."
        rm -rf $TEMP_RESULTS
    fi
fi

# 清理Python缓存文件
echo "清理Python缓存文件..."
find "$PROJECT_DIR" -name "*.pyc" -delete
find "$PROJECT_DIR" -name "*.pyo" -delete
find "$PROJECT_DIR" -name ".DS_Store" -delete

echo ""
echo "清理完成！"