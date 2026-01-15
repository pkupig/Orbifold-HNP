"""
结果处理器 - 完整实现
"""
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from ..core.graph_builder import GraphBuilder, UnitDistanceGraph
from ..core.geometry_engine import GeometryEngine
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class ResultHandler:
    """结果处理器"""
    
    def __init__(self, results_dir: str = "results"):
        """
        初始化结果处理器
        
        Args:
            results_dir: 结果目录
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Result handler initialized with directory: {results_dir}")
    
    def load_result(self, experiment_name: str, 
                   load_graph: bool = True) -> Dict[str, Any]:
        """
        加载实验结果
        
        Args:
            experiment_name: 实验名称
            load_graph: 是否加载图数据
            
        Returns:
            实验结果字典
        """
        exp_dir = self.results_dir / experiment_name
        
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
        
        result = {}
        
        # 加载结果摘要
        result_file = exp_dir / "result.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                result.update(json.load(f))
        else:
            logger.warning(f"No result.json found in {exp_dir}")
        
        # 加载图
        if load_graph:
            graph_file = exp_dir / "graph.pkl"
            if graph_file.exists():
                try:
                    graph = GraphBuilder.load_graph(str(graph_file))
                    result['graph'] = graph
                    logger.debug(f"Graph loaded: {graph.num_nodes} nodes")
                except Exception as e:
                    logger.error(f"Failed to load graph from {graph_file}: {e}")
            else:
                logger.debug(f"No graph file found in {exp_dir}")
        
        # 加载历史
        history_file = exp_dir / "history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                result['history'] = json.load(f)
        
        # 加载配置
        config_file = exp_dir / "config.yaml"
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                result['config'] = yaml.safe_load(f)
        
        result['experiment_name'] = experiment_name
        result['experiment_dir'] = str(exp_dir)
        
        return result
    
    def analyze_graph(self, graph: UnitDistanceGraph, 
                     geometry: Optional[GeometryEngine] = None) -> Dict[str, Any]:
        """
        分析图的结构属性
        
        Args:
            graph: 单位距离图
            geometry: 几何引擎（可选）
            
        Returns:
            图分析结果
        """
        if graph.num_nodes == 0:
            return {"empty_graph": True}
        
        # 转换为NetworkX图
        nx_graph = graph.to_networkx()
        
        # 基本统计
        analysis = {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "edge_density": graph.edge_density,
            "average_degree": np.mean(list(dict(nx_graph.degree()).values())),
            "degree_std": np.std(list(dict(nx_graph.degree()).values())),
            "min_degree": min(dict(nx_graph.degree()).values()),
            "max_degree": max(dict(nx_graph.degree()).values()),
        }
        
        # 连通性分析
        if nx.is_connected(nx_graph):
            analysis["is_connected"] = True
            try:
                analysis["diameter"] = nx.diameter(nx_graph)
                analysis["average_path_length"] = nx.average_shortest_path_length(nx_graph)
            except:
                analysis["diameter"] = None
                analysis["average_path_length"] = None
        else:
            analysis["is_connected"] = False
            components = list(nx.connected_components(nx_graph))
            analysis["num_components"] = len(components)
            analysis["component_sizes"] = [len(c) for c in components]
            analysis["largest_component_size"] = max(analysis["component_sizes"])
        
        # 聚类系数
        try:
            analysis["average_clustering"] = nx.average_clustering(nx_graph)
        except:
            analysis["average_clustering"] = 0.0
        
        # 如果提供了几何引擎，计算几何属性
        if geometry is not None:
            # 计算距离矩阵（样本）
            n_samples = min(100, graph.num_nodes)
            if n_samples > 1:
                sample_indices = np.random.choice(graph.num_nodes, n_samples, replace=False)
                sample_nodes = graph.nodes[sample_indices]
                
                # 计算样本间的距离
                distances = []
                for i in range(n_samples):
                    for j in range(i+1, n_samples):
                        dist = geometry.get_metric(sample_nodes[i], sample_nodes[j])
                        distances.append(dist)
                
                if distances:
                    analysis["sample_distance_mean"] = np.mean(distances)
                    analysis["sample_distance_std"] = np.std(distances)
                    analysis["sample_distance_min"] = min(distances)
                    analysis["sample_distance_max"] = max(distances)
        
        # 度分布
        degree_dist = dict(nx_graph.degree())
        analysis["degree_distribution"] = degree_dist
        
        # 计算度分布的统计
        degrees = list(degree_dist.values())
        analysis["degree_histogram"] = np.histogram(degrees, bins=min(10, max(degrees)+1))[0].tolist()
        
        return analysis
    
    def compare_experiments(self, experiment_names: List[str], 
                          include_graph_analysis: bool = False) -> pd.DataFrame:
        """
        比较多个实验
        
        Args:
            experiment_names: 实验名称列表
            include_graph_analysis: 是否包含图分析
            
        Returns:
            比较结果的DataFrame
        """
        rows = []
        
        for name in experiment_names:
            try:
                result = self.load_result(name, load_graph=include_graph_analysis)
                
                row = {
                    "experiment": name,
                    "success": result.get("success", False),
                    "chromatic_number": result.get("chromatic_number", -1),
                    "nodes": result.get("final_graph_size", [0, 0])[0],
                    "edges": result.get("final_graph_size", [0, 0])[1],
                    "iterations": result.get("iterations", 0),
                    "runtime": result.get("runtime", 0.0),
                    "termination_reason": result.get("termination_reason", "unknown")
                }
                
                # 添加配置信息
                config = result.get("config", {})
                if "geometry" in config:
                    row["lattice_type"] = config["geometry"].get("lattice_type", "unknown")
                if "pipeline" in config:
                    row["target_k"] = config["pipeline"].get("target_k", 0)
                
                # 可选：图分析
                if include_graph_analysis and "graph" in result:
                    geometry_config = config.get("geometry", {})
                    geometry = GeometryEngine(
                        lattice_type=geometry_config.get("lattice_type", "hexagonal")
                    )
                    
                    graph_analysis = self.analyze_graph(result["graph"], geometry)
                    
                    # 添加关键图分析指标
                    row["edge_density"] = graph_analysis.get("edge_density", 0.0)
                    row["average_degree"] = graph_analysis.get("average_degree", 0.0)
                    row["is_connected"] = graph_analysis.get("is_connected", False)
                    row["average_clustering"] = graph_analysis.get("average_clustering", 0.0)
                
                rows.append(row)
                
            except Exception as e:
                logger.error(f"Failed to process experiment {name}: {e}")
                rows.append({
                    "experiment": name,
                    "error": str(e)
                })
        
        # 创建DataFrame
        df = pd.DataFrame(rows)
        
        # 如果DataFrame不为空，计算一些统计
        if not df.empty and "chromatic_number" in df.columns:
            success_mask = df["success"] == True
            if success_mask.any():
                success_df = df[success_mask]
                logger.info(f"Successful experiments: {len(success_df)}")
                logger.info(f"Average chromatic number: {success_df['chromatic_number'].mean():.2f}")
                logger.info(f"Average runtime: {success_df['runtime'].mean():.2f}s")
        
        return df
    
    def export_to_csv(self, experiment_names: List[str], 
                     output_file: str = "experiment_comparison.csv"):
        """
        导出实验结果到CSV
        
        Args:
            experiment_names: 实验名称列表
            output_file: 输出文件路径
            
        Returns:
            DataFrame
        """
        df = self.compare_experiments(experiment_names, include_graph_analysis=True)
        
        if not df.empty:
            df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(df)} experiments to {output_file}")
        
        return df
    
    def generate_comparison_plot(self, experiment_names: List[str],
                               output_file: Optional[str] = None):
        """
        生成实验比较图
        
        Args:
            experiment_names: 实验名称列表
            output_file: 输出文件路径（可选）
        """
        df = self.compare_experiments(experiment_names)
        
        if df.empty or "chromatic_number" not in df.columns:
            logger.warning("No valid data for comparison plot")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Experiment Comparison", fontsize=16)
        
        # 1. 色数分布
        ax = axes[0, 0]
        chromatic_numbers = df["chromatic_number"]
        ax.hist(chromatic_numbers[chromatic_numbers > 0], bins=range(2, 12), 
                edgecolor='black', alpha=0.7)
        ax.set_xlabel("Chromatic Number")
        ax.set_ylabel("Count")
        ax.set_title("Chromatic Number Distribution")
        ax.grid(True, alpha=0.3)
        
        # 2. 运行时间 vs 色数
        ax = axes[0, 1]
        success_mask = df["success"] == True
        ax.scatter(df.loc[success_mask, "chromatic_number"], 
                  df.loc[success_mask, "runtime"], 
                  alpha=0.6, label="Success")
        ax.scatter(df.loc[~success_mask, "chromatic_number"], 
                  df.loc[~success_mask, "runtime"], 
                  alpha=0.6, label="Failed", marker='x')
        ax.set_xlabel("Chromatic Number")
        ax.set_ylabel("Runtime (s)")
        ax.set_title("Runtime vs Chromatic Number")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 节点数 vs 边数
        ax = axes[0, 2]
        ax.scatter(df["nodes"], df["edges"], alpha=0.6, 
                  c=df["chromatic_number"], cmap="viridis")
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Number of Edges")
        ax.set_title("Graph Size")
        ax.grid(True, alpha=0.3)
        
        # 4. 迭代次数分布
        ax = axes[1, 0]
        ax.hist(df["iterations"], bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Count")
        ax.set_title("Iteration Distribution")
        ax.grid(True, alpha=0.3)
        
        # 5. 成功/失败比例
        ax = axes[1, 1]
        success_count = df["success"].sum()
        fail_count = len(df) - success_count
        ax.pie([success_count, fail_count], 
               labels=["Success", "Failure"],
               autopct='%1.1f%%', startangle=90)
        ax.set_title("Success Rate")
        
        # 6. 晶格类型对比
        ax = axes[1, 2]
        if "lattice_type" in df.columns:
            lattice_counts = df["lattice_type"].value_counts()
            ax.bar(lattice_counts.index, lattice_counts.values)
            ax.set_xlabel("Lattice Type")
            ax.set_ylabel("Count")
            ax.set_title("Lattice Type Distribution")
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {output_file}")
        
        plt.show()
        return fig
    
    def analyze_trends(self, experiment_pattern: str = "experiment_*") -> Dict[str, Any]:
        """
        分析实验趋势
        
        Args:
            experiment_pattern: 实验名称模式
            
        Returns:
            趋势分析结果
        """
        # 查找匹配的实验
        exp_dirs = list(self.results_dir.glob(experiment_pattern))
        experiment_names = [d.name for d in exp_dirs if d.is_dir()]
        
        if not experiment_names:
            return {"error": "No experiments found"}
        
        # 加载所有实验结果
        all_results = []
        for name in experiment_names:
            try:
                result = self.load_result(name, load_graph=False)
                all_results.append(result)
            except:
                continue
        
        if not all_results:
            return {"error": "No valid results found"}
        
        # 按时间排序（假设实验名称包含时间戳或按创建时间排序）
        all_results.sort(key=lambda x: x.get("runtime", 0))
        
        # 提取趋势数据
        trends = {
            "experiment_count": len(all_results),
            "chromatic_numbers": [r.get("chromatic_number", 0) for r in all_results],
            "graph_sizes": [r.get("final_graph_size", [0, 0]) for r in all_results],
            "runtimes": [r.get("runtime", 0) for r in all_results],
            "iterations": [r.get("iterations", 0) for r in all_results],
            "success_rates": [r.get("success", False) for r in all_results]
        }
        
        # 计算统计
        if trends["chromatic_numbers"]:
            trends["avg_chromatic"] = np.mean(trends["chromatic_numbers"])
            trends["max_chromatic"] = max(trends["chromatic_numbers"])
            trends["min_chromatic"] = min(trends["chromatic_numbers"])
        
        if trends["runtimes"]:
            trends["avg_runtime"] = np.mean(trends["runtimes"])
            trends["total_runtime"] = sum(trends["runtimes"])
        
        trends["success_count"] = sum(trends["success_rates"])
        trends["success_rate"] = trends["success_count"] / len(all_results) if all_results else 0
        
        return trends
    
    def create_summary_report(self, output_file: str = "summary_report.md"):
        """
        创建总结报告（Markdown格式）
        
        Args:
            output_file: 输出文件路径
        """
        # 获取所有实验
        exp_dirs = list(self.results_dir.glob("*"))
        experiment_names = [d.name for d in exp_dirs if d.is_dir() and (d / "result.json").exists()]
        
        if not experiment_names:
            report = "# Experiment Summary Report\n\nNo experiments found."
            with open(output_file, 'w') as f:
                f.write(report)
            return
        
        # 加载并分析所有实验
        df = self.compare_experiments(experiment_names)
        
        # 生成Markdown报告
        report = f"""# Experiment Summary Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Experiments:** {len(experiment_names)}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Successful Experiments | {df['success'].sum()} |
| Average Chromatic Number | {df[df['chromatic_number'] > 0]['chromatic_number'].mean():.2f} |
| Average Runtime | {df['runtime'].mean():.2f} seconds |
| Average Iterations | {df['iterations'].mean():.1f} |
| Average Graph Size | {df['nodes'].mean():.1f} nodes, {df['edges'].mean():.1f} edges |

## Top Experiments by Chromatic Number

"""
        
        # 按色数排序
        top_experiments = df.sort_values("chromatic_number", ascending=False).head(10)
        
        report += "| Experiment | χ | Nodes | Edges | Runtime | Success |\n"
        report += "|------------|---|-------|-------|---------|---------|\n"
        
        for _, row in top_experiments.iterrows():
            success_symbol = "✓" if row["success"] else "✗"
            report += f"| {row['experiment']} | {row['chromatic_number']} | "
            report += f"{row['nodes']} | {row['edges']} | {row['runtime']:.1f}s | {success_symbol} |\n"
        
        report += "\n## Detailed Results\n\n"
        
        # 详细结果表格
        report += "| Experiment | Lattice | Target k | χ | Nodes | Edges | Iterations | Runtime | Status |\n"
        report += "|------------|---------|----------|---|-------|-------|------------|---------|--------|\n"
        
        for _, row in df.iterrows():
            status = "Success" if row["success"] else "Failed"
            lattice = row.get("lattice_type", "unknown")
            target_k = row.get("target_k", "N/A")
            
            report += f"| {row['experiment']} | {lattice} | {target_k} | "
            report += f"{row['chromatic_number']} | {row['nodes']} | {row['edges']} | "
            report += f"{row['iterations']} | {row['runtime']:.1f}s | {status} |\n"
        
        # 保存报告
        with open(output_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Summary report saved to {output_file}")
        
        return report

# 快捷函数
def analyze_experiment(experiment_name: str, results_dir: str = "results") -> Dict[str, Any]:
    """分析单个实验的快捷函数"""
    handler = ResultHandler(results_dir)
    result = handler.load_result(experiment_name, load_graph=True)
    
    if "graph" in result:
        geometry_config = result.get("config", {}).get("geometry", {})
        geometry = GeometryEngine(
            lattice_type=geometry_config.get("lattice_type", "hexagonal")
        )
        
        analysis = handler.analyze_graph(result["graph"], geometry)
        result["graph_analysis"] = analysis
    
    return result

def analyze_experiment(experiment_name: str, results_dir: str = "results") -> Dict[str, Any]:
    """分析单个实验的快捷函数"""
    handler = ResultHandler(results_dir)
    return handler.load_result(experiment_name)

def compare_experiments(experiment_names: Optional[List[str]] = None, 
                       results_dir: str = "results") -> pd.DataFrame:
    """
    比较多个实验的快捷函数
    
    Args:
        experiment_names: 实验名称列表，None表示比较所有
        results_dir: 结果目录
        
    Returns:
        包含实验对比数据的DataFrame
    """
    handler = ResultHandler(results_dir)
    return handler.get_summary_dataframe(experiment_names)

def export_results(output_file: str, 
                  experiment_names: Optional[List[str]] = None,
                  results_dir: str = "results",
                  format: str = "csv"):
    """
    导出实验结果
    
    Args:
        output_file: 输出文件路径
        experiment_names: 实验名称列表
        results_dir: 结果目录
        format: 导出格式 (csv, json, excel)
    """
    df = compare_experiments(experiment_names, results_dir)
    
    if format == "csv":
        df.to_csv(output_file, index=False)
    elif format == "json":
        df.to_json(output_file, orient="records", indent=2)
    elif format == "excel":
        df.to_excel(output_file, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Results exported to {output_file}")

    