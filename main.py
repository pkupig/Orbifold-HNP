#!/usr/bin/env python3
"""
轨形染色系统 - 主入口点
"""
import argparse
import sys
import os
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.coloring_pipeline import OrbifoldColoringPipeline
from utils.logging_config import setup_logging

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Orbifold-based Unit Distance Graph Coloring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s run --config config/hexagonal.yaml --experiment test1
  %(prog)s visualize --graph data/graphs/test1.pkl
  %(prog)s estimate --graph data/graphs/test1.pkl --max-k 6
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # run命令：运行管道
    run_parser = subparsers.add_parser("run", help="运行染色管道")
    run_parser.add_argument("--config", type=str, default="config/default_config.yaml",
                          help="配置文件路径")
    run_parser.add_argument("--experiment", type=str, default="experiment_001",
                          help="实验名称")
    run_parser.add_argument("--output", type=str, default="results",
                          help="输出目录")
    
    # visualize命令：可视化图
    vis_parser = subparsers.add_parser("visualize", help="可视化图")
    vis_parser.add_argument("--graph", type=str, required=True,
                          help="图文件路径")
    vis_parser.add_argument("--show-cover", action="store_true",
                          help="显示覆盖空间")
    vis_parser.add_argument("--output", type=str,
                          help="输出图像路径")
    
    # estimate命令：估计色数
    est_parser = subparsers.add_parser("estimate", help="估计图色数")
    est_parser.add_argument("--graph", type=str, required=True,
                          help="图文件路径")
    est_parser.add_argument("--max-k", type=int, default=6,
                          help="最大测试颜色数")
    est_parser.add_argument("--timeout", type=int, default=30,
                          help="每个测试的超时时间")
    
    # test命令：运行测试
    test_parser = subparsers.add_parser("test", help="运行测试")
    test_parser.add_argument("--test-case", type=str,
                           help="特定测试用例")
    test_parser.add_argument("--verbose", action="store_true",
                           help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    if args.command == "run":
        run_pipeline(args)
    elif args.command == "visualize":
        visualize_graph(args)
    elif args.command == "estimate":
        estimate_chromatic_number(args)
    elif args.command == "test":
        run_tests(args)
    else:
        parser.print_help()
        sys.exit(1)

def run_pipeline(args):
    """运行管道"""
    print(f"运行实验: {args.experiment}")
    print(f"配置文件: {args.config}")
    
    # 创建管道
    pipeline = OrbifoldColoringPipeline(config_path=args.config)
    
    # 运行管道
    result = pipeline.run(experiment_name=args.experiment)
    
    # 打印结果
    print("\n" + "="*60)
    print("实验结果:")
    print(f"  成功: {result.success}")
    print(f"  估计色数: {result.chromatic_number}")
    print(f"  最终图大小: {result.final_graph_size[0]} 节点, {result.final_graph_size[1]} 边")
    print(f"  迭代次数: {result.iterations}")
    print("="*60)
    
    # 保存结果
    output_dir = Path(args.output) / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / "result.json"
    import json
    with open(result_file, 'w') as f:
        json.dump({
            "experiment": args.experiment,
            "success": result.success,
            "chromatic_number": result.chromatic_number,
            "final_graph_size": result.final_graph_size,
            "iterations": result.iterations,
            "config": result.config
        }, f, indent=2)
    
    print(f"结果保存到: {result_file}")

def visualize_graph(args):
    """可视化图"""
    from visualization.graph_visualizer import GraphVisualizer
    from src.core.graph_builder import GraphBuilder
    from src.core.geometry_engine import GeometryEngine
    
    print(f"加载图: {args.graph}")
    
    # 加载图
    graph = GraphBuilder.load_graph(args.graph)
    
    # 创建几何引擎（假设为六边形晶格）
    geometry = GeometryEngine()
    
    # 可视化
    visualizer = GraphVisualizer()
    
    if args.show_cover:
        fig = visualizer.plot_cover(graph, geometry)
    else:
        fig = visualizer.plot_graph(graph, geometry)
    
    if args.output:
        fig.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"图像保存到: {args.output}")
    
    import matplotlib.pyplot as plt
    plt.show()

def estimate_chromatic_number(args):
    """估计色数"""
    from src.core.graph_builder import GraphBuilder
    from src.core.sat_solver import SATSolver
    
    print(f"加载图: {args.graph}")
    
    # 加载图
    graph = GraphBuilder.load_graph(args.graph)
    
    print(f"图信息: {graph.num_nodes} 节点, {graph.num_edges} 边, epsilon={graph.epsilon}")
    
    # 创建SAT求解器
    solver = SATSolver()
    
    # 估计色数
    chromatic_num, stats = solver.estimate_chromatic_number(
        graph, 
        max_k=args.max_k,
        timeout_per_test=args.timeout
    )
    
    print("\n" + "="*60)
    print(f"估计色数: {chromatic_num}")
    print("="*60)
    
    # 打印详细统计
    print("\n测试统计:")
    for test in stats["tests"]:
        k = test["k"]
        colorable = test["colorable"]
        time = test["time"]
        
        if colorable is None:
            status = "超时"
        elif colorable:
            status = "可染色"
        else:
            status = "不可染色"
        
        print(f"  {k}-染色: {status} ({time:.1f}秒)")

def run_tests(args):
    """运行测试"""
    import pytest
    
    test_args = ["tests/"]
    
    if args.test_case:
        test_args.append(f"tests/test_{args.test_case}.py")
    
    if args.verbose:
        test_args.append("-v")
    
    # 运行pytest
    sys.exit(pytest.main(test_args))

if __name__ == "__main__":
    main()