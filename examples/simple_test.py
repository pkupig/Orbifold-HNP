"""
简单测试示例 
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

# --- 修改处：导入 LatticeConfig ---
from src.core.geometry_engine import GeometryEngine, LatticeConfig
from src.core.graph_builder import GraphBuilder, GraphConfig
from src.core.sat_solver import SATSolver, SATConfig
from src.visualization.graph_visualizer import GraphVisualizer

def main():
    """主函数：简单测试示例"""
    print("=" * 60)
    print("简单测试：构建小型单位距离图并测试染色")
    print("=" * 60)
    
    # 1. 创建几何引擎（使用正方形晶格简化）
    print("\n1. 创建几何引擎...")
    # --- 修改处：使用配置对象初始化 ---
    geometry = GeometryEngine(LatticeConfig(type="square"))
    print(f"   晶格类型: {geometry.config.type}")
    print(f"   基础向量: v1={geometry.v1}, v2={geometry.v2}")
    
    # 2. 创建图构建器
    print("\n2. 创建图构建器...")
    graph_config = GraphConfig(epsilon=0.05, use_kdtree=False)
    builder = GraphBuilder(geometry, graph_config)
    
    # 3. 创建简单图（正方形加对角线）
    print("\n3. 构建简单图...")
    nodes = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [0.0, 1.0],  # 2
        [1.0, 1.0],  # 3
        [0.5, 0.0],  # 4 - 中点
    ])
    
    # 构建图
    graph = builder.construct_graph(nodes, use_kdtree=False)
    print(f"   节点数: {graph.num_nodes}")
    print(f"   边数: {graph.num_edges}")
    print(f"   Epsilon: {graph.epsilon}")
    
    # 4. 创建SAT求解器
    print("\n4. 创建SAT求解器...")
    # 优先使用 kissat，如果没有则使用 minisat
    try:
        sat_config = SATConfig(solver_name="kissat", timeout=10)
        solver = SATSolver(sat_config)
        print("   使用求解器: kissat")
    except:
        sat_config = SATConfig(solver_name="minisat", timeout=10)
        solver = SATSolver(sat_config)
        print("   使用求解器: minisat")
    
    # 5. 测试不同颜色数的染色性
    print("\n5. 测试染色性:")
    
    for k in range(2, 6):
        colorable, coloring, stats = solver.is_k_colorable(graph, k)
        
        if colorable is None:
            print(f"   {k}-染色: 超时")
        elif colorable:
            print(f"   {k}-染色: 是")
            print(f"     使用的颜色: {len(set(coloring.values()))}")
        else:
            print(f"   {k}-染色: 否")
    
    # 6. 估计色数
    print("\n6. 估计色数...")
    chromatic_num, stats = solver.estimate_chromatic_number(graph, max_k=5)
    print(f"   估计色数: {chromatic_num}")
    
    # 7. 可视化
    print("\n7. 可视化...")
    visualizer = GraphVisualizer()
    
    # 获取3-染色方案（如果存在）
    colorable, coloring, _ = solver.is_k_colorable(graph, 3)
    
    if colorable and coloring:
        print(f"   显示3-染色方案")
        fig = visualizer.plot_graph(
            graph, geometry, 
            coloring=coloring,
            title=f"Simple Test Graph (3-colorable)",
            show_labels=True
        )
    else:
        fig = visualizer.plot_graph(
            graph, geometry,
            title="Simple Test Graph",
            show_labels=True
        )
    
    # 保存图像而不是只显示，方便在远程环境查看
    fig.savefig("simple_test_result.png")
    print("   图像已保存至 simple_test_result.png")
    
    # 8. 覆盖空间可视化
    print("\n8. 覆盖空间可视化...")
    fig2 = visualizer.plot_cover(
        graph, geometry,
        title="Cover Space Visualization"
    )
    fig2.savefig("cover_space_result.png")
    print("   图像已保存至 cover_space_result.png")
    
    print("\n" + "=" * 60)
    print("简单测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()