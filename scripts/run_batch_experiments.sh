#!/bin/bash
# 批量运行实验脚本

echo "批量运行轨形染色实验"
echo "====================="

# 设置变量
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
RESULTS_DIR="$PROJECT_DIR/results"
EXPERIMENTS_DIR="$PROJECT_DIR/experiments"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建目录
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$EXPERIMENTS_DIR"

echo "项目目录: $PROJECT_DIR"
echo "结果目录: $RESULTS_DIR"
echo "实验目录: $EXPERIMENTS_DIR"
echo "日志目录: $LOG_DIR"
echo "时间戳: $TIMESTAMP"
echo ""

# 检查实验配置
if [ ! -d "$EXPERIMENTS_DIR" ] || [ -z "$(ls -A "$EXPERIMENTS_DIR"/*.yaml 2>/dev/null)" ]; then
    echo "警告: 实验目录为空或不存在!"
    echo "创建示例实验配置..."
    
    # 创建示例配置
    cat > "$EXPERIMENTS_DIR/example_hexagonal.yaml" << EOF
geometry:
  lattice_type: "hexagonal"
  v1: [1.0, 0.0]
  v2: [0.5, 0.86602540378]

graph_builder:
  epsilon: 0.02
  sampling_method: "fibonacci"

pipeline:
  initial_points: 20
  target_k: 4
  max_iterations: 10

description: "示例实验 - 六边形晶格"
enabled: true
priority: 1
EOF
    
    cat > "$EXPERIMENTS_DIR/example_square.yaml" << EOF
geometry:
  lattice_type: "square"
  v1: [1.0, 0.0]
  v2: [0.0, 1.0]

graph_builder:
  epsilon: 0.02
  sampling_method: "grid"

pipeline:
  initial_points: 16  # 完全平方数
  target_k: 4
  max_iterations: 10

description: "示例实验 - 正方形晶格"
enabled: true
priority: 2
EOF
    
    echo "创建了示例实验配置"
fi

# 设置日志文件
LOG_FILE="$LOG_DIR/batch_experiment_$TIMESTAMP.log"

echo "开始批量实验..."
echo "日志文件: $LOG_FILE"
echo ""

# 运行批量实验
cd "$PROJECT_DIR"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3"
    exit 1
fi

# 运行实验
echo "运行实验..." | tee -a "$LOG_FILE"
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')

from src.pipeline.experiment_runner import run_experiment_batch

print('开始批量实验...')
report = run_experiment_batch(
    experiments_dir='$EXPERIMENTS_DIR',
    max_workers=1,  # 顺序执行
    report_file='$RESULTS_DIR/experiment_report_$TIMESTAMP.json'
)
print('批量实验完成！')
print(f'报告保存到: $RESULTS_DIR/experiment_report_$TIMESTAMP.json')
" 2>&1 | tee -a "$LOG_FILE"

# 生成报告
echo ""
echo "生成实验报告..." | tee -a "$LOG_FILE"

python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')

from src.visualization.results_analyzer import ResultsAnalyzer

analyzer = ResultsAnalyzer('$RESULTS_DIR')

# 生成比较图
fig = analyzer.create_comparison_plot(
    output_file='$RESULTS_DIR/comparison_$TIMESTAMP.png'
)

# 生成详细报告
report = analyzer.generate_report(
    output_file='$RESULTS_DIR/analysis_report_$TIMESTAMP.md'
)

print('报告生成完成！')
print(f'比较图: $RESULTS_DIR/comparison_$TIMESTAMP.png')
print(f'分析报告: $RESULTS_DIR/analysis_report_$TIMESTAMP.md')
" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "批量实验完成！"
echo "结果保存在: $RESULTS_DIR"
echo "日志文件: $LOG_FILE"