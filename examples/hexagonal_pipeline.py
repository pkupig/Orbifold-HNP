"""
六边形管道示例 - 完整实现
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.coloring_pipeline import OrbifoldColoringPipeline
from src.utils.logging_config import setup_default_logging

def main():
    """主函数：六边形管道示例"""
    print("=" * 80)
    print("六边形晶格管道示例")
    print("=" * 80)
    
    # 设置日志
    log_file = setup_default_logging()
    print(f"日志文件: {log_file}")
    
    # 1. 创建管道
    print("\n1. 创建管道...")
    config_path = os.path.join(
        os.path.dirname(__file__), 
        "../config/hexagonal_config.yaml"
    )
    
    pipeline = OrbifoldColoringPipeline(config_path)
    print(f"   配置文件: {config_path}")
    
    # 2. 运行管道
    print("\n2. 运行管道...")
    result = pipeline.run(
        experiment_name="hexagonal_example",
        max_iterations=15
    )
    
    # 3. 显示结果
    print("\n3. 结果:")
    print(f"   成功找到反例: {result.success}")
    print(f"   估计色数: {result.chromatic_number}")
    print(f"   最终图大小: {result.final_graph_size[0]} 节点, {result.final_graph_size[1]} 边")
    print(f"   迭代次数: {result.iterations}")
    print(f"   运行时间: {result.runtime:.2f} 秒")
    print(f"   终止原因: {result.termination_reason}")
    
    # 4. 获取统计信息
    print("\n4. 统计信息:")
    stats = pipeline.get_statistics()
    
    print(f"   SAT求解器统计:")
    solver_stats = stats.get("solver_stats", {})
    print(f"     总求解次数: {solver_stats.get('total_solves', 0)}")
    print(f"     可满足: {solver_stats.get('satisfiable', 0)}")
    print(f"     不可满足: {solver_stats.get('unsatisfiable', 0)}")
    print(f"     超时: {solver_stats.get('timeouts', 0)}")
    
    print(f"   优化器统计:")
    optimizer_stats = stats.get("optimizer_history", {})
    print(f"     总操作次数: {optimizer_stats.get('total_operations', 0)}")
    print(f"     生成的点数: {optimizer_stats.get('points_generated', 0)}")
    
    # 5. 可视化当前状态
    print("\n5. 可视化...")
    pipeline.visualize_current_state(
        show_cover=True
    )
    
    print("\n" + "=" * 80)
    print("六边形管道示例完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()