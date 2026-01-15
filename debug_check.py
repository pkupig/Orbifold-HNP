"""
诊断脚本：直接检查实验结果文件
"""
import pickle
import json
import os
import sys
import numpy as np
from pathlib import Path

# 尝试导入项目模块（如果需要）
sys.path.insert(0, os.getcwd())

def inspect_results(experiment_name):
    print(f"\n{'='*50}")
    print(f"诊断实验: {experiment_name}")
    print(f"{'='*50}")
    
    base_dir = Path("results") / experiment_name
    
    if not base_dir.exists():
        print(f"❌ 目录不存在: {base_dir}")
        return

    # 1. 检查 result.json
    json_path = base_dir / "result.json"
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # 检查关键字段
            pipeline_res = data.get("pipeline_result", {})
            # 兼容不同层级的结构
            if not pipeline_res and "final_graph_size" in data:
                pipeline_res = data
            
            g_size = pipeline_res.get("final_graph_size", ["N/A", "N/A"])
            success = pipeline_res.get("success", "Unknown")
            
            print(f"✅ result.json 读取成功")
            print(f"   - 记录的图规模: {g_size}")
            print(f"   - 实验成功状态: {success}")
            print(f"   - 记录的迭代数: {pipeline_res.get('iterations', '?')}")
        except Exception as e:
            print(f"❌ result.json 读取失败: {e}")
    else:
        print(f"❌ result.json 缺失")

    # 2. 检查 graph.pkl
    pkl_path = base_dir / "graph.pkl"
    if pkl_path.exists():
        print(f"\n🔍 检查 graph.pkl ({pkl_path.stat().st_size} bytes)...")
        try:
            with open(pkl_path, 'rb') as f:
                graph_data = pickle.load(f)
            
            print(f"✅ graph.pkl 读取成功 (类型: {type(graph_data)})")
            
            nodes = None
            edges = None
            
            # 可能是字典，也可能是对象
            if isinstance(graph_data, dict):
                print(f"   - 数据是字典格式")
                nodes = graph_data.get("nodes")
                edges = graph_data.get("edges")
                epsilon = graph_data.get("epsilon")
                print(f"   - Epsilon: {epsilon}")
            else:
                print(f"   - 数据是对象格式: {type(graph_data)}")
                if hasattr(graph_data, "nodes"):
                    nodes = graph_data.nodes
                if hasattr(graph_data, "edges"):
                    edges = graph_data.edges
            
            # 统计实际数据
            n_nodes = len(nodes) if nodes is not None else 0
            n_edges = len(edges) if edges is not None else 0
            
            print(f"   - 实际节点数: {n_nodes}")
            print(f"   - 实际边数: {n_edges}")
            
            if n_nodes > 0 and n_edges == 0:
                print("\n⚠️  诊断: 图有节点但没有边。")
                print("   原因: Epsilon 太小，或者点分布太稀疏，无法在环面上形成连接。")
            elif n_nodes == 0:
                print("\n⚠️  诊断: 图为空。")
                print("   原因: 初始化点生成失败，或保存逻辑有误。")
                
        except Exception as e:
            print(f"❌ graph.pkl 读取崩溃: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ graph.pkl 缺失")

if __name__ == "__main__":
    # 检查刚才运行的两个实验
    inspect_results("dense_300_k4")
    inspect_results("dense_500_k5")