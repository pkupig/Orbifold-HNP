"""
完整实验示例 - 修复版 (关闭 KDTree 以支持周期性边界)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import yaml
from pathlib import Path

from src.pipeline.experiment_runner import run_experiment_batch
from src.visualization.results_analyzer import ResultsAnalyzer

def main():
    """主函数：修复图生成逻辑的实验"""
    print("=" * 80)
    print("完整实验示例 - 周期性边界修复版 (Disable KDTree)")
    print("=" * 80)
    
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(exist_ok=True)
    
    # ---------------------------------------------------------
    # 实验 1: 300点 + 暴力搜索 (Brute Force)
    # ---------------------------------------------------------
    exp1_config = {
        "pipeline_config": {
            "geometry": {
                "lattice_type": "hexagonal",
                "v1": [1.0, 0.0],
                "v2": [0.5, 0.86602540378]
            },
            "graph_builder": {
                # [关键修复] 关闭 KDTree！
                # KDTree 不支持环面(Torus)的周期性距离，会导致所有跨边界的边丢失。
                "use_kdtree": False,  
                "epsilon": 0.05,
                "sampling_method": "fibonacci"
            },
            "optimizer": {
                "method": "hybrid",
                "num_candidates": 1000,
                "relaxation_iterations": 10
            },
            "pipeline": {
                "initial_points": 300,
                "target_k": 4,
                "max_iterations": 15,
                "anneal_epsilon": True,
                "epsilon_decay": 0.96, # 缓慢衰减
                "min_epsilon": 0.03,   # 保持较大容差
                "save_interval": 3
            }
        },
        "description": "Fix: Brute force distance check (No KDTree)",
        "enabled": True,
        "priority": 1
    }
    
    # ---------------------------------------------------------
    # 实验 2: 500点 + 高密度
    # ---------------------------------------------------------
    exp2_config = {
        "pipeline_config": {
            "geometry": {
                "lattice_type": "hexagonal",
                "v1": [1.0, 0.0],
                "v2": [0.5, 0.86602540378]
            },
            "graph_builder": {
                "use_kdtree": False,  # [关键修复]
                "epsilon": 0.04,
                "sampling_method": "grid" # 网格采样
            },
            "optimizer": {
                "method": "hybrid",
                "num_candidates": 2000,
                "relaxation_iterations": 15
            },
            "pipeline": {
                "initial_points": 500,
                "target_k": 5,
                "max_iterations": 20,
                "anneal_epsilon": True,
                "min_epsilon": 0.025
            }
        },
        "description": "Fix: High density with brute force",
        "enabled": True,
        "priority": 2
    }
    
    # 保存配置
    configs = [exp1_config, exp2_config]
    names = ["fix_300_k4", "fix_500_k5"]
    
    for name, config in zip(names, configs):
        path = experiments_dir / f"{name}.yaml"
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"   已创建配置: {name}")
    
    # 运行
    print("\n3. 运行批量实验...")
    run_experiment_batch(
        experiments_dir="experiments",
        max_workers=1,
        report_file="fix_experiment_report.json"
    )
    
    # 分析
    print("\n4. 分析结果...")
    analyzer = ResultsAnalyzer(results_dir="results")
    
    try:
        for name in names:
            res = analyzer.load_experiment(name)
            if res:
                g_size = res.get('final_graph_size', [0, 0])
                k_est = res.get('chromatic_number', '?')
                nodes = g_size[0]
                edges = g_size[1]
                
                print(f"\n实验 [{name}]:")
                print(f"  - 最终规模: {nodes} 点, {edges} 边")
                
                if nodes > 0:
                    density = 2 * edges / (nodes * (nodes - 1))
                    print(f"  - 边密度: {density:.4f}")
                    avg_degree = 2 * edges / nodes
                    print(f"  - 平均度数: {avg_degree:.2f}")
                
                print(f"  - 估计色数: {k_est}")
                
                if edges > 0:
                    print("  ✅ 成功！生成了有效的单位距离图。")
                else:
                    print("  ❌ 依然没有边。")
                    
    except Exception as e:
        print(f"分析时发生错误: {e}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()