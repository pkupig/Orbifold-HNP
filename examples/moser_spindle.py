"""
Moser spindle示例 - 完整实现 (已修复)
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

def create_moser_spindle() -> np.ndarray:
    """创建Moser spindle的坐标"""
    nodes = []
    # 基础菱形 (A, B, C, D)
    nodes.append(np.array([0.0, 0.0]))
    nodes.append(np.array([1.0, 0.0]))
    angle_60 = np.pi / 3
    nodes.append(np.array([0.5, np.sin(angle_60)]))
    nodes.append(np.array([1.5, np.sin(angle_60)]))
    
    # 点E，与A和C距离为1
    angle_120 = 2 * np.pi / 3
    nodes.append(np.array([0.5 + np.cos(angle_120), np.sin(angle_60) + np.sin(angle_120)]))
    # 点F，与B和D距离为1
    nodes.append(np.array([1.5 + np.cos(angle_120), np.sin(angle_60) + np.sin(angle_120)]))
    # 点G，与E和F距离为1
    nodes.append(np.array([1.0, 2 * np.sin(angle_60)]))
    
    return np.array(nodes)

def main():
    """主函数：Moser spindle示例"""
    print("=" * 60)
    print("Moser Spindle 示例")
    print("=" * 60)
    
    # 1. 创建几何引擎
    print("\n1. 创建几何引擎...")
    # --- 修改处：使用配置对象初始化 ---
    geometry = GeometryEngine(LatticeConfig(type="hexagonal"))
    print(f"   晶格类型: {geometry.config.type}")
    
    # 2. 创建Moser spindle
    print("\n2. 创建Moser spindle...")
    nodes = create_moser_spindle()
    print(f"   节点数: {len(nodes)}")
    
    # 3. 创建图构建器
    print("\n3. 创建图构建器...")
    graph_config = GraphConfig(epsilon=0.01, use_kdtree=False)
    builder = GraphBuilder(geometry, graph_config)
    graph = builder.construct_graph(nodes, use_kdtree=False)
    print(f"   边数: {graph.num_edges}")
    
    # 4. 创建SAT求解器
    print("\n4. 创建SAT求解器...")
    try:
        sat_config = SATConfig(solver_name="kissat", timeout=30)
        solver = SATSolver(sat_config)
    except:
        sat_config = SATConfig(solver_name="minisat", timeout=30)
        solver = SATSolver(sat_config)
    
    # 5. 测试染色性
    print("\n5. 测试染色性:")
    coloring_4 = None
    for k in [2, 3, 4, 5]:
        colorable, coloring, stats = solver.is_k_colorable(graph, k)
        if colorable is None:
            print(f"   {k}-染色: 超时")
        elif colorable:
            print(f"   {k}-染色: 是")
            if k == 4: coloring_4 = coloring
        else:
            print(f"   {k}-染色: 否")
    
    # ... (其余部分保持不变，建议添加 fig.savefig) ...
    visualizer = GraphVisualizer()
    if coloring_4:
        fig1 = visualizer.plot_graph(graph, geometry, coloring=coloring_4, title="Moser Spindle")
        fig1.savefig("moser_spindle.png")
        print("   已保存 moser_spindle.png")

if __name__ == "__main__":
    main()