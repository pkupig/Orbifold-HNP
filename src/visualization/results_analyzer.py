"""
结果分析器 - 完整实现
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from ..core.graph_builder import UnitDistanceGraph
from ..core.geometry_engine import GeometryEngine
from ..utils.data_io import load_results
from ..utils.metrics import graph_metrics, coloring_metrics, pipeline_metrics

class ResultsAnalyzer:
    """结果分析器"""
    
    def __init__(self, results_dir: str = "results"):
        """
        初始化结果分析器
        
        Args:
            results_dir: 结果目录
        """
        self.results_dir = Path(results_dir)
        self.results_cache = {}
        
        # 确保目录存在
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_experiment(self, experiment_name: str, 
                       force_reload: bool = False) -> Dict[str, Any]:
        """
        加载实验
        
        Args:
            experiment_name: 实验名称
            force_reload: 是否强制重新加载
            
        Returns:
            实验数据
        """
        if not force_reload and experiment_name in self.results_cache:
            return self.results_cache[experiment_name]
        
        try:
            result = load_results(str(self.results_dir), experiment_name)
            self.results_cache[experiment_name] = result
            return result
        except Exception as e:
            print(f"Failed to load experiment {experiment_name}: {e}")
            return {}
    
    def list_experiments(self, pattern: str = "*") -> List[str]:
        """
        列出所有实验
        
        Args:
            pattern: 匹配模式
            
        Returns:
            实验名称列表
        """
        exp_dirs = list(self.results_dir.glob(pattern))
        return [d.name for d in exp_dirs if d.is_dir() and (d / "result.json").exists()]
    
    def compare_experiments(self, experiment_names: Optional[List[str]] = None,
                           include_graph_analysis: bool = False) -> pd.DataFrame:
        """
        比较多个实验
        
        Args:
            experiment_names: 实验名称列表（None表示所有实验）
            include_graph_analysis: 是否包含图分析
            
        Returns:
            比较结果的DataFrame
        """
        if experiment_names is None:
            experiment_names = self.list_experiments()
        
        rows = []
        
        for name in experiment_names:
            try:
                result = self.load_experiment(name)
                
                if not result:
                    continue
                
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
                    row["initial_points"] = config["pipeline"].get("initial_points", 0)
                
                # 可选：图分析
                if include_graph_analysis and "graph" in result:
                    geometry_config = config.get("geometry", {})
                    geometry = GeometryEngine(
                        lattice_type=geometry_config.get("lattice_type", "hexagonal")
                    )
                    
                    graph_analysis = graph_metrics(result["graph"])
                    
                    # 添加关键图分析指标
                    row["edge_density"] = graph_analysis.get("basic", {}).get("edge_density", 0.0)
                    row["average_degree"] = graph_analysis.get("degree", {}).get("mean", 0.0)
                    row["is_connected"] = graph_analysis.get("connectivity", {}).get("is_connected", False)
                    row["average_clustering"] = graph_analysis.get("clustering", {}).get("average", 0.0)
                
                rows.append(row)
                
            except Exception as e:
                print(f"Failed to process experiment {name}: {e}")
                rows.append({
                    "experiment": name,
                    "error": str(e)
                })
        
        # 创建DataFrame
        df = pd.DataFrame(rows)
        
        return df
    
    def analyze_trends(self, experiment_pattern: str = "experiment_*") -> Dict[str, Any]:
        """
        分析实验趋势
        
        Args:
            experiment_pattern: 实验名称模式
            
        Returns:
            趋势分析结果
        """
        experiment_names = self.list_experiments(experiment_pattern)
        
        if not experiment_names:
            return {"error": "No experiments found"}
        
        df = self.compare_experiments(experiment_names)
        
        if df.empty:
            return {"error": "No valid data"}
        
        # 分析趋势
        trends = {
            "experiment_count": len(df),
            "success_rate": df["success"].mean(),
            "average_chromatic_number": df[df["chromatic_number"] > 0]["chromatic_number"].mean(),
            "average_runtime": df["runtime"].mean(),
            "average_iterations": df["iterations"].mean(),
            "size_trend": {
                "nodes": df["nodes"].mean(),
                "edges": df["edges"].mean()
            }
        }
        
        # 按晶格类型分组
        if "lattice_type" in df.columns:
            lattice_groups = df.groupby("lattice_type")
            
            lattice_stats = {}
            for lattice, group in lattice_groups:
                lattice_stats[lattice] = {
                    "count": len(group),
                    "success_rate": group["success"].mean(),
                    "avg_chromatic": group[group["chromatic_number"] > 0]["chromatic_number"].mean(),
                    "avg_runtime": group["runtime"].mean()
                }
            
            trends["by_lattice"] = lattice_stats
        
        # 按目标k分组
        if "target_k" in df.columns:
            k_groups = df.groupby("target_k")
            
            k_stats = {}
            for k, group in k_groups:
                k_stats[k] = {
                    "count": len(group),
                    "success_rate": group["success"].mean(),
                    "avg_chromatic": group[group["chromatic_number"] > 0]["chromatic_number"].mean()
                }
            
            trends["by_target_k"] = k_stats
        
        return trends
    
    def create_comparison_plot(self, experiment_names: Optional[List[str]] = None,
                              output_file: Optional[str] = None) -> plt.Figure:
        """
        创建实验比较图
        
        Args:
            experiment_names: 实验名称列表
            output_file: 输出文件路径
            
        Returns:
            matplotlib图形对象
        """
        if experiment_names is None:
            experiment_names = self.list_experiments()
        
        df = self.compare_experiments(experiment_names)
        
        if df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No data available", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Experiment Comparison", fontsize=16)
        
        # 1. 色数分布
        ax = axes[0, 0]
        chromatic_numbers = df["chromatic_number"]
        valid_chromatic = chromatic_numbers[chromatic_numbers > 0]
        
        if len(valid_chromatic) > 0:
            ax.hist(valid_chromatic, bins=range(2, 12), 
                   edgecolor='black', alpha=0.7)
            ax.set_xlabel("Chromatic Number")
            ax.set_ylabel("Count")
            ax.set_title("Chromatic Number Distribution")
        else:
            ax.text(0.5, 0.5, "No chromatic number data", 
                   ha='center', va='center', fontsize=12)
        
        ax.grid(True, alpha=0.3)
        
        # 2. 运行时间 vs 色数
        ax = axes[0, 1]
        success_mask = df["success"] == True
        
        if success_mask.any():
            success_data = df[success_mask]
            if "chromatic_number" in success_data.columns:
                ax.scatter(success_data["chromatic_number"], 
                          success_data["runtime"], 
                          alpha=0.6, label="Success", color='green')
        
        if (~success_mask).any():
            fail_data = df[~success_mask]
            if "chromatic_number" in fail_data.columns:
                ax.scatter(fail_data["chromatic_number"], 
                          fail_data["runtime"], 
                          alpha=0.6, label="Failed", color='red', marker='x')
        
        ax.set_xlabel("Chromatic Number")
        ax.set_ylabel("Runtime (s)")
        ax.set_title("Runtime vs Chromatic Number")
        
        if success_mask.any() or (~success_mask).any():
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        
        # 3. 节点数 vs 边数
        ax = axes[0, 2]
        
        if "nodes" in df.columns and "edges" in df.columns:
            scatter = ax.scatter(df["nodes"], df["edges"], 
                               alpha=0.6, c=df["chromatic_number"], 
                               cmap="viridis")
            
            ax.set_xlabel("Number of Nodes")
            ax.set_ylabel("Number of Edges")
            ax.set_title("Graph Size")
            
            # 添加颜色条
            plt.colorbar(scatter, ax=ax, label="Chromatic Number")
        else:
            ax.text(0.5, 0.5, "No graph size data", 
                   ha='center', va='center', fontsize=12)
        
        ax.grid(True, alpha=0.3)
        
        # 4. 迭代次数分布
        ax = axes[1, 0]
        
        if "iterations" in df.columns:
            ax.hist(df["iterations"], bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Count")
            ax.set_title("Iteration Distribution")
        else:
            ax.text(0.5, 0.5, "No iteration data", 
                   ha='center', va='center', fontsize=12)
        
        ax.grid(True, alpha=0.3)
        
        # 5. 成功/失败比例
        ax = axes[1, 1]
        
        if "success" in df.columns:
            success_count = df["success"].sum()
            fail_count = len(df) - success_count
            
            ax.pie([success_count, fail_count], 
                  labels=["Success", "Failure"],
                  autopct='%1.1f%%', startangle=90,
                  colors=['green', 'red'])
            ax.set_title("Success Rate")
        else:
            ax.text(0.5, 0.5, "No success data", 
                   ha='center', va='center', fontsize=12)
        
        # 6. 晶格类型对比
        ax = axes[1, 2]
        
        if "lattice_type" in df.columns:
            lattice_counts = df["lattice_type"].value_counts()
            
            if not lattice_counts.empty:
                bars = ax.bar(range(len(lattice_counts)), lattice_counts.values)
                ax.set_xticks(range(len(lattice_counts)))
                ax.set_xticklabels(lattice_counts.index, rotation=45)
                ax.set_xlabel("Lattice Type")
                ax.set_ylabel("Count")
                ax.set_title("Lattice Type Distribution")
            else:
                ax.text(0.5, 0.5, "No lattice type data", 
                       ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, "No lattice type data", 
                   ha='center', va='center', fontsize=12)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {output_file}")
        
        return fig
    
    def generate_report(self, experiment_names: Optional[List[str]] = None,
                       output_file: str = "analysis_report.md") -> str:
        """
        生成分析报告（Markdown格式）
        
        Args:
            experiment_names: 实验名称列表
            output_file: 输出文件路径
            
        Returns:
            报告字符串
        """
        if experiment_names is None:
            experiment_names = self.list_experiments()
        
        if not experiment_names:
            report = "# Analysis Report\n\nNo experiments found."
            with open(output_file, 'w') as f:
                f.write(report)
            return report
        
        df = self.compare_experiments(experiment_names)
        
        # 计算统计
        total_experiments = len(df)
        successful = df["success"].sum()
        failed = total_experiments - successful
        success_rate = successful / total_experiments if total_experiments > 0 else 0
        
        avg_chromatic = df[df["chromatic_number"] > 0]["chromatic_number"].mean()
        avg_runtime = df["runtime"].mean()
        avg_iterations = df["iterations"].mean()
        
        # 生成报告
        import datetime
        
        report = f"""# Experiment Analysis Report

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Experiments:** {total_experiments}
**Successful:** {successful}
**Failed:** {failed}
**Success Rate:** {success_rate:.1%}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Average Chromatic Number | {avg_chromatic:.2f} |
| Average Runtime | {avg_runtime:.2f} seconds |
| Average Iterations | {avg_iterations:.1f} |
| Average Graph Size | {df["nodes"].mean():.1f} nodes, {df["edges"].mean():.1f} edges |

## Top Experiments by Chromatic Number

"""
        
        # 按色数排序
        top_experiments = df.sort_values("chromatic_number", ascending=False).head(10)
        
        report += "| Experiment | χ | Nodes | Edges | Runtime | Success | Lattice |\n"
        report += "|------------|---|-------|-------|---------|---------|---------|\n"
        
        for _, row in top_experiments.iterrows():
            success_symbol = "✓" if row["success"] else "✗"
            lattice = row.get("lattice_type", "unknown")
            
            report += f"| {row['experiment']} | {row['chromatic_number']} | "
            report += f"{row['nodes']} | {row['edges']} | {row['runtime']:.1f}s | "
            report += f"{success_symbol} | {lattice} |\n"
        
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
        
        # 添加趋势分析
        report += "\n## Trend Analysis\n\n"
        
        # 按晶格类型分组
        if "lattice_type" in df.columns:
            lattice_groups = df.groupby("lattice_type")
            
            report += "### By Lattice Type\n\n"
            for lattice, group in lattice_groups:
                group_success = group["success"].sum()
                group_total = len(group)
                group_rate = group_success / group_total if group_total > 0 else 0
                
                report += f"- **{lattice}**: {group_total} experiments, "
                report += f"success rate: {group_rate:.1%}, "
                report += f"avg χ: {group[group['chromatic_number'] > 0]['chromatic_number'].mean():.2f}\n"
            
            report += "\n"
        
        # 按目标k分组
        if "target_k" in df.columns:
            k_groups = df.groupby("target_k")
            
            report += "### By Target k\n\n"
            for k, group in k_groups:
                group_success = group["success"].sum()
                group_total = len(group)
                group_rate = group_success / group_total if group_total > 0 else 0
                
                report += f"- **k = {k}**: {group_total} experiments, "
                report += f"success rate: {group_rate:.1%}, "
                report += f"avg χ: {group[group['chromatic_number'] > 0]['chromatic_number'].mean():.2f}\n"
        
        # 保存报告
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Analysis report saved to {output_file}")
        
        return report
    
    def find_best_experiment(self, metric: str = "chromatic_number") -> Dict[str, Any]:
        """
        根据指标找到最佳实验
        
        Args:
            metric: 指标名称 ("chromatic_number", "runtime", "nodes", "edges")
            
        Returns:
            最佳实验信息
        """
        experiment_names = self.list_experiments()
        
        if not experiment_names:
            return {"error": "No experiments found"}
        
        df = self.compare_experiments(experiment_names)
        
        if df.empty:
            return {"error": "No valid data"}
        
        # 根据指标选择最佳实验
        if metric == "chromatic_number":
            # 寻找色数最高的成功实验
            successful = df[df["success"] == True]
            if not successful.empty:
                best_idx = successful["chromatic_number"].idxmax()
            else:
                best_idx = df["chromatic_number"].idxmax()
        elif metric == "runtime":
            # 寻找运行时间最短的实验（可能不够有意义）
            best_idx = df["runtime"].idxmin()
        elif metric == "nodes":
            # 寻找节点最多的实验
            best_idx = df["nodes"].idxmax()
        elif metric == "edges":
            # 寻找边最多的实验
            best_idx = df["edges"].idxmax()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        best_experiment = df.loc[best_idx].to_dict()
        
        # 加载完整结果
        full_result = self.load_experiment(best_experiment["experiment"])
        best_experiment["full_result"] = full_result
        
        return best_experiment