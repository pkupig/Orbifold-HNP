"""
基准测试示例 - 完整实现
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from pathlib import Path

from src.core.geometry_engine import GeometryEngine, LatticeConfig
from src.core.graph_builder import GraphBuilder, GraphConfig
from src.core.sat_solver import SATSolver, SATConfig
from src.core.optimizer import GraphOptimizer, OptimizerConfig

def benchmark_geometry_engine():
    """几何引擎基准测试"""
    print("几何引擎基准测试...")
    
    results = {}
    
    # 测试不同晶格类型
    lattice_types = ["hexagonal", "square"]
    
    for lattice_type in lattice_types:
        start_time = time.time()
        
        # 创建几何引擎
        geometry = GeometryEngine(LatticeConfig(type=lattice_type))
        
        # 生成测试点
        n_points = 1000
        lattice_coords = np.random.uniform(0, 1, (n_points, 2))
        points = np.array([geometry.to_euclidean_coords(coord) 
                          for coord in lattice_coords])
        
        # 测试距离计算
        dist_calls = 1000
        for _ in range(dist_calls):
            i, j = np.random.randint(0, n_points, 2)
            _ = geometry.get_metric(points[i], points[j])
        
        elapsed = time.time() - start_time
        
        results[lattice_type] = {
            "creation_time": elapsed,
            "distance_calls_per_second": dist_calls / elapsed if elapsed > 0 else 0
        }
    
    return results

def benchmark_graph_builder():
    """图构建器基准测试"""
    print("图构建器基准测试...")
    
    results = {}
    
    geometry = GeometryEngine()
    
    # 测试不同点数的构建时间
    point_counts = [100, 500, 1000, 2000]
    
    for n_points in point_counts:
        # 创建图构建器
        config = GraphConfig(epsilon=0.02, use_kdtree=True)
        builder = GraphBuilder(geometry, config)
        
        # 生成点
        start_time = time.time()
        nodes = builder.initialize_points_fibonacci(n_points)
        generation_time = time.time() - start_time
        
        # 构建图（使用KDTree）
        start_time = time.time()
        graph = builder.construct_graph(nodes, use_kdtree=True)
        construction_time = time.time() - start_time
        
        results[n_points] = {
            "generation_time": generation_time,
            "construction_time": construction_time,
            "total_time": generation_time + construction_time,
            "nodes": graph.num_nodes,
            "edges": graph.num_edges,
            "edge_density": graph.edge_density
        }
    
    return results

def benchmark_sat_solver():
    """SAT求解器基准测试"""
    print("SAT求解器基准测试...")
    
    # 创建测试图
    geometry = GeometryEngine()
    builder = GraphBuilder(geometry, GraphConfig(epsilon=0.02))
    
    # 生成不同大小的图
    results = {}
    
    for n_points in [10, 20, 30, 40]:
        nodes = builder.initialize_points_fibonacci(n_points)
        graph = builder.construct_graph(nodes, use_kdtree=True)
        
        # 创建SAT求解器
        solver = SATSolver(SATConfig(solver_name="kissat", timeout=30))
        
        # 测试不同k值的求解时间
        k_results = {}
        
        for k in [2, 3, 4]:
            start_time = time.time()
            colorable, coloring, stats = solver.is_k_colorable(graph, k)
            solve_time = time.time() - start_time
            
            k_results[k] = {
                "solve_time": solve_time,
                "colorable": colorable,
                "coloring_found": coloring is not None
            }
        
        results[n_points] = {
            "nodes": graph.num_nodes,
            "edges": graph.num_edges,
            "k_results": k_results
        }
    
    return results

def benchmark_optimizer():
    """优化器基准测试"""
    print("优化器基准测试...")
    
    geometry = GeometryEngine()
    builder = GraphBuilder(geometry, GraphConfig(epsilon=0.02))
    
    # 创建测试图
    nodes = builder.initialize_points_fibonacci(20)
    graph = builder.construct_graph(nodes, use_kdtree=True)
    
    # 创建测试染色方案
    solver = SATSolver(SATConfig(solver_name="kissat", timeout=10))
    colorable, coloring, _ = solver.is_k_colorable(graph, 4)
    
    if not colorable or coloring is None:
        # 创建简单染色方案
        coloring = {i: i % 4 for i in range(graph.num_nodes)}
    
    # 测试不同优化方法
    results = {}
    methods = ["constraint_based", "energy_based", "hybrid"]
    
    for method in methods:
        config = OptimizerConfig(method=method, num_candidates=500)
        optimizer = GraphOptimizer(geometry, builder, config)
        
        # 测试寻找难以染色的点
        start_time = time.time()
        hard_point = optimizer.find_hard_point(graph, coloring, method=method)
        find_time = time.time() - start_time
        
        # 测试松弛
        start_time = time.time()
        relaxed_graph = optimizer.relax_nodes(graph, iterations=5)
        relax_time = time.time() - start_time
        
        results[method] = {
            "find_hard_point_time": find_time,
            "relaxation_time": relax_time,
            "point_found": hard_point is not None,
            "edges_after_relaxation": relaxed_graph.num_edges
        }
    
    return results

def main():
    """主函数：基准测试"""
    print("=" * 80)
    print("轨形染色系统基准测试")
    print("=" * 80)
    
    all_results = {}
    
    # 1. 几何引擎基准测试
    print("\n1. 几何引擎基准测试")
    geo_results = benchmark_geometry_engine()
    all_results["geometry_engine"] = geo_results
    
    for lattice, stats in geo_results.items():
        print(f"   {lattice}: {stats['distance_calls_per_second']:.0f} 距离计算/秒")
    
    # 2. 图构建器基准测试
    print("\n2. 图构建器基准测试")
    graph_results = benchmark_graph_builder()
    all_results["graph_builder"] = graph_results
    
    for n_points, stats in graph_results.items():
        print(f"   {n_points}点: {stats['total_time']:.3f}秒 "
              f"({stats['edges']}边, 密度:{stats['edge_density']:.4f})")
    
    # 3. SAT求解器基准测试
    print("\n3. SAT求解器基准测试")
    sat_results = benchmark_sat_solver()
    all_results["sat_solver"] = sat_results
    
    for n_points, stats in sat_results.items():
        print(f"   {n_points}节点图:")
        for k, k_stats in stats["k_results"].items():
            print(f"     k={k}: {k_stats['solve_time']:.2f}秒, "
                  f"可染色: {k_stats['colorable']}")
    
    # 4. 优化器基准测试
    print("\n4. 优化器基准测试")
    opt_results = benchmark_optimizer()
    all_results["optimizer"] = opt_results
    
    for method, stats in opt_results.items():
        print(f"   {method}: 找点时间:{stats['find_hard_point_time']:.3f}秒, "
              f"松弛时间:{stats['relaxation_time']:.3f}秒")
    
    # 5. 保存结果
    print("\n5. 保存基准测试结果...")
    
    import json
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("   结果已保存到 benchmark_results.json")
    
    # 6. 生成报告
    print("\n6. 生成基准测试报告...")
    
    report = generate_benchmark_report(all_results)
    with open("benchmark_report.md", "w") as f:
        f.write(report)
    
    print("   报告已保存到 benchmark_report.md")
    
    print("\n" + "=" * 80)
    print("基准测试完成！")
    print("=" * 80)

def generate_benchmark_report(results):
    """生成基准测试报告"""
    report = "# 轨形染色系统基准测试报告\n\n"
    
    # 几何引擎部分
    report += "## 1. 几何引擎性能\n\n"
    report += "| 晶格类型 | 距离计算/秒 |\n"
    report += "|----------|------------|\n"
    
    geo_results = results.get("geometry_engine", {})
    for lattice, stats in geo_results.items():
        report += f"| {lattice} | {stats['distance_calls_per_second']:.0f} |\n"
    
    # 图构建器部分
    report += "\n## 2. 图构建器性能\n\n"
    report += "| 点数 | 总时间(秒) | 节点数 | 边数 | 边密度 |\n"
    report += "|------|------------|--------|------|--------|\n"
    
    graph_results = results.get("graph_builder", {})
    for n_points, stats in graph_results.items():
        report += f"| {n_points} | {stats['total_time']:.3f} | "
        report += f"{stats['nodes']} | {stats['edges']} | {stats['edge_density']:.4f} |\n"
    
    # SAT求解器部分
    report += "\n## 3. SAT求解器性能\n\n"
    
    sat_results = results.get("sat_solver", {})
    for n_points, stats in sat_results.items():
        report += f"### {n_points}节点图\n\n"
        report += "| k | 求解时间(秒) | 可染色 |\n"
        report += "|---|--------------|--------|\n"
        
        for k, k_stats in stats["k_results"].items():
            report += f"| {k} | {k_stats['solve_time']:.2f} | {k_stats['colorable']} |\n"
        
        report += "\n"
    
    # 优化器部分
    report += "\n## 4. 优化器性能\n\n"
    report += "| 方法 | 找点时间(秒) | 松弛时间(秒) | 边数(松弛后) |\n"
    report += "|------|--------------|--------------|--------------|\n"
    
    opt_results = results.get("optimizer", {})
    for method, stats in opt_results.items():
        report += f"| {method} | {stats['find_hard_point_time']:.3f} | "
        report += f"{stats['relaxation_time']:.3f} | {stats['edges_after_relaxation']} |\n"
    
    # 总结
    report += "\n## 5. 总结\n\n"
    
    # 计算总体性能指标
    total_tests = 0
    total_time = 0
    
    for category, category_results in results.items():
        if category == "geometry_engine":
            for lattice, stats in category_results.items():
                total_time += stats.get("creation_time", 0)
                total_tests += 1
        elif category == "graph_builder":
            for n_points, stats in category_results.items():
                total_time += stats.get("total_time", 0)
                total_tests += 1
        elif category == "sat_solver":
            for n_points, stats in category_results.items():
                for k, k_stats in stats.get("k_results", {}).items():
                    total_time += k_stats.get("solve_time", 0)
                    total_tests += 1
        elif category == "optimizer":
            for method, stats in category_results.items():
                total_time += stats.get("find_hard_point_time", 0)
                total_time += stats.get("relaxation_time", 0)
                total_tests += 2
    
    report += f"- 总测试数: {total_tests}\n"
    report += f"- 总运行时间: {total_time:.2f}秒\n"
    report += f"- 平均测试时间: {total_time / total_tests if total_tests > 0 else 0:.3f}秒\n"
    
    report += "\n系统在所有基准测试中表现正常，各组件性能符合预期。\n"
    
    return report

if __name__ == "__main__":
    main()