"""
指标计算工具 - 完整实现
计算图、染色方案和管道的各种指标
"""
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional, Set
from scipy import stats, spatial
import math

from ..core.graph_builder import UnitDistanceGraph
from ..core.geometry_engine import GeometryEngine

def graph_metrics(graph: UnitDistanceGraph, 
                  geometry: Optional[GeometryEngine] = None) -> Dict[str, Any]:
    """
    计算图的结构和几何指标
    
    Args:
        graph: 单位距离图
        geometry: 几何引擎（用于计算几何指标）
        
    Returns:
        指标字典
    """
    if graph.num_nodes == 0:
        return {"empty": True}
    
    nx_graph = graph.to_networkx()
    
    metrics = {
        "basic": {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "edge_density": graph.edge_density,
            "epsilon": graph.epsilon
        }
    }
    
    # 度统计
    degrees = list(dict(nx_graph.degree()).values())
    metrics["degree"] = {
        "mean": float(np.mean(degrees)),
        "std": float(np.std(degrees)),
        "min": int(np.min(degrees)),
        "max": int(np.max(degrees)),
        "median": float(np.median(degrees)),
        "skewness": float(stats.skew(degrees) if len(degrees) > 1 else 0),
        "kurtosis": float(stats.kurtosis(degrees) if len(degrees) > 1 else 0)
    }
    
    # 度分布直方图
    if degrees:
        hist, bins = np.histogram(degrees, bins=min(10, max(degrees) + 1))
        metrics["degree"]["histogram"] = {
            "counts": hist.tolist(),
            "bins": bins.tolist()
        }
    
    # 连通性指标
    if nx.is_connected(nx_graph):
        metrics["connectivity"] = {
            "is_connected": True,
            "diameter": nx.diameter(nx_graph),
            "average_path_length": nx.average_shortest_path_length(nx_graph),
            "radius": nx.radius(nx_graph),
            "center_size": len(nx.center(nx_graph)),
            "periphery_size": len(nx.periphery(nx_graph))
        }
    else:
        components = list(nx.connected_components(nx_graph))
        component_sizes = [len(c) for c in components]
        metrics["connectivity"] = {
            "is_connected": False,
            "num_components": len(components),
            "component_sizes": component_sizes,
            "largest_component": max(component_sizes),
            "smallest_component": min(component_sizes) if component_sizes else 0,
            "avg_component_size": np.mean(component_sizes) if component_sizes else 0
        }
    
    # 聚类系数
    try:
        clustering = nx.clustering(nx_graph)
        clustering_values = list(clustering.values())
        metrics["clustering"] = {
            "average": nx.average_clustering(nx_graph),
            "global": nx.transitivity(nx_graph),
            "min": min(clustering_values) if clustering_values else 0,
            "max": max(clustering_values) if clustering_values else 0,
            "distribution": np.histogram(clustering_values, bins=10)[0].tolist()
        }
    except:
        metrics["clustering"] = {"average": 0.0, "global": 0.0}
    
    # 中心性指标（采样计算）
    n_samples = min(50, graph.num_nodes)
    if n_samples > 0:
        sample_nodes = list(nx_graph.nodes())[:n_samples]
        
        # 度中心性
        degree_centrality = nx.degree_centrality(nx_graph)
        sampled_degree_centrality = [degree_centrality[node] for node in sample_nodes]
        
        # 接近中心性（如果图连通）
        if nx.is_connected(nx_graph):
            closeness_centrality = nx.closeness_centrality(nx_graph)
            sampled_closeness = [closeness_centrality[node] for node in sample_nodes]
        else:
            sampled_closeness = [0.0] * n_samples
        
        metrics["centrality"] = {
            "degree_mean": float(np.mean(sampled_degree_centrality)),
            "degree_std": float(np.std(sampled_degree_centrality)),
            "closeness_mean": float(np.mean(sampled_closeness)) if sampled_closeness else 0.0,
            "closeness_std": float(np.std(sampled_closeness)) if sampled_closeness else 0.0
        }
    
    # 图的谱特性（拉普拉斯矩阵的特征值）
    try:
        laplacian = nx.laplacian_matrix(nx_graph).todense()
        eigenvalues = np.linalg.eigvals(laplacian)
        eigenvalues_real = eigenvalues.real
        eigenvalues_real.sort()
        
        metrics["spectral"] = {
            "algebraic_connectivity": float(eigenvalues_real[1]),  # Fiedler值
            "largest_eigenvalue": float(eigenvalues_real[-1]),
            "eigenvalue_ratio": float(eigenvalues_real[-1] / max(eigenvalues_real[1], 1e-10)),
            "eigenvalue_gap": float(eigenvalues_real[1] - eigenvalues_real[0])
        }
    except:
        metrics["spectral"] = {"error": "Failed to compute spectral properties"}
    
    # 几何指标（如果提供了几何引擎）
    if geometry is not None and graph.num_nodes > 1:
        metrics["geometry"] = _compute_geometry_metrics(graph, geometry)
    
    return metrics

def _compute_geometry_metrics(graph: UnitDistanceGraph, 
                            geometry: GeometryEngine) -> Dict[str, Any]:
    """计算几何相关指标"""
    nodes = graph.nodes
    
    # 基本几何统计
    centroid = np.mean(nodes, axis=0)
    distances_to_centroid = [np.linalg.norm(node - centroid) for node in nodes]
    
    # 边界框
    min_coords = np.min(nodes, axis=0)
    max_coords = np.max(nodes, axis=0)
    bbox_size = max_coords - min_coords
    bbox_area = bbox_size[0] * bbox_size[1]
    
    # 距离分布（采样）
    n_samples = min(100, graph.num_nodes)
    if n_samples > 1:
        sample_indices = np.random.choice(graph.num_nodes, n_samples, replace=False)
        sample_nodes = nodes[sample_indices]
        
        distances = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = geometry.get_metric(sample_nodes[i], sample_nodes[j])
                distances.append(dist)
        
        if distances:
            metrics = {
                "centroid": centroid.tolist(),
                "centroid_distance_mean": float(np.mean(distances_to_centroid)),
                "centroid_distance_std": float(np.std(distances_to_centroid)),
                "bounding_box": {
                    "min": min_coords.tolist(),
                    "max": max_coords.tolist(),
                    "size": bbox_size.tolist(),
                    "area": float(bbox_area)
                },
                "distance_distribution": {
                    "mean": float(np.mean(distances)),
                    "std": float(np.std(distances)),
                    "min": float(np.min(distances)),
                    "max": float(np.max(distances)),
                    "median": float(np.median(distances))
                }
            }
            
            # 检查单位距离边的比例
            unit_distances = [d for d in distances if abs(d - 1.0) < graph.epsilon]
            if distances:
                metrics["unit_distance_ratio"] = len(unit_distances) / len(distances)
            
            return metrics
    
    return {}

def coloring_metrics(graph: UnitDistanceGraph, 
                    coloring: Dict[int, int],
                    detailed: bool = False) -> Dict[str, Any]:
    """
    计算染色方案的指标
    
    Args:
        graph: 单位距离图
        coloring: 染色方案
        detailed: 是否计算详细指标
        
    Returns:
        染色指标字典
    """
    if not coloring:
        return {"empty": True}
    
    nx_graph = graph.to_networkx()
    
    # 检查染色有效性
    valid = True
    conflicts = []
    
    for u, v in graph.edges:
        if u in coloring and v in coloring:
            if coloring[u] == coloring[v]:
                valid = False
                conflicts.append((u, v))
    
    # 颜色统计
    color_values = list(coloring.values())
    unique_colors = set(color_values)
    num_colors = len(unique_colors)
    
    # 颜色分布
    color_counts = {color: color_values.count(color) for color in unique_colors}
    color_proportions = {color: count / len(coloring) for color, count in color_counts.items()}
    
    metrics = {
        "valid": valid,
        "conflicts": len(conflicts),
        "conflict_edges": conflicts if detailed else [],
        "num_colors": num_colors,
        "color_distribution": color_counts,
        "color_proportions": color_proportions
    }
    
    # 平衡性指标
    if color_counts:
        counts = list(color_counts.values())
        metrics["balance"] = {
            "min_count": min(counts),
            "max_count": max(counts),
            "entropy": float(stats.entropy(counts)),
            "gini": _gini_coefficient(counts),
            "uniformity": min(counts) / max(counts) if max(counts) > 0 else 0
        }
    
    # 如果染色有效，计算更多指标
    if valid and detailed:
        # 计算每个节点的邻居颜色分布
        neighbor_color_stats = []
        
        for node in nx_graph.nodes():
            if node in coloring:
                node_color = coloring[node]
                neighbors = list(nx_graph.neighbors(node))
                
                neighbor_colors = []
                for neighbor in neighbors:
                    if neighbor in coloring:
                        neighbor_colors.append(coloring[neighbor])
                
                if neighbor_colors:
                    same_color_count = neighbor_colors.count(node_color)
                    different_color_count = len(neighbor_colors) - same_color_count
                    
                    neighbor_color_stats.append({
                        "node": node,
                        "same_color_neighbors": same_color_count,
                        "different_color_neighbors": different_color_count,
                        "total_neighbors": len(neighbor_colors),
                        "same_color_ratio": same_color_count / len(neighbor_colors) if neighbor_colors else 0
                    })
        
        if neighbor_color_stats:
            same_color_ratios = [s["same_color_ratio"] for s in neighbor_color_stats]
            
            metrics["neighborhood"] = {
                "avg_same_color_ratio": float(np.mean(same_color_ratios)),
                "max_same_color_ratio": float(np.max(same_color_ratios)),
                "min_same_color_ratio": float(np.min(same_color_ratios)),
                "std_same_color_ratio": float(np.std(same_color_ratios))
            }
        
        # 计算颜色类间的边数
        color_classes = {}
        for node, color in coloring.items():
            color_classes.setdefault(color, []).append(node)
        
        inter_class_edges = {}
        intra_class_edges = {}
        
        for color, nodes in color_classes.items():
            # 类内边
            class_graph = nx_graph.subgraph(nodes)
            intra_class_edges[color] = class_graph.number_of_edges()
            
            # 类间边
            inter_class_edges[color] = 0
            for node in nodes:
                neighbors = list(nx_graph.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in nodes:
                        inter_class_edges[color] += 1
        
        metrics["class_edges"] = {
            "intra_class": intra_class_edges,
            "inter_class": inter_class_edges,
            "total_intra": sum(intra_class_edges.values()),
            "total_inter": sum(inter_class_edges.values()) // 2  # 每条边被计数两次
        }
    
    return metrics

def _gini_coefficient(values: List[float]) -> float:
    """计算基尼系数"""
    if not values:
        return 0.0
    
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    
    if np.sum(values) == 0:
        return 0.0
    
    return (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))

def pipeline_metrics(history: List[Dict[str, Any]], 
                    include_detailed: bool = False) -> Dict[str, Any]:
    """
    计算管道运行指标
    
    Args:
        history: 管道历史记录
        include_detailed: 是否包含详细指标
        
    Returns:
        管道指标字典
    """
    if not history:
        return {"empty": True}
    
    # 提取关键数据
    iterations = []
    node_counts = []
    edge_counts = []
    epsilons = []
    actions = []
    
    for entry in history:
        if "iteration" in entry:
            iterations.append(entry["iteration"])
        
        if "nodes" in entry:
            node_counts.append(entry["nodes"])
        
        if "edges" in entry:
            edge_counts.append(entry["edges"])
        
        if "epsilon" in entry:
            epsilons.append(entry["epsilon"])
        
        if "action" in entry:
            actions.append(entry["action"])
    
    metrics = {
        "iterations": {
            "total": len(iterations),
            "max": max(iterations) if iterations else 0,
            "actions": {
                action: actions.count(action) for action in set(actions)
            }
        },
        "growth": {
            "initial_nodes": node_counts[0] if node_counts else 0,
            "final_nodes": node_counts[-1] if node_counts else 0,
            "initial_edges": edge_counts[0] if edge_counts else 0,
            "final_edges": edge_counts[-1] if edge_counts else 0,
            "node_growth": node_counts[-1] - node_counts[0] if node_counts else 0,
            "edge_growth": edge_counts[-1] - edge_counts[0] if edge_counts else 0,
            "avg_node_growth": np.mean(np.diff(node_counts)) if len(node_counts) > 1 else 0,
            "avg_edge_growth": np.mean(np.diff(edge_counts)) if len(edge_counts) > 1 else 0
        },
        "density": {
            "initial": edge_counts[0] / (node_counts[0] * (node_counts[0] - 1) / 2) 
                     if node_counts and node_counts[0] > 1 else 0,
            "final": edge_counts[-1] / (node_counts[-1] * (node_counts[-1] - 1) / 2)
                    if node_counts and node_counts[-1] > 1 else 0,
            "change": 0  # 将在下面计算
        }
    }
    
    # 计算密度变化
    if metrics["density"]["initial"] > 0:
        metrics["density"]["change"] = (
            metrics["density"]["final"] - metrics["density"]["initial"]
        ) / metrics["density"]["initial"]
    
    # Epsilon退火指标
    if epsilons and len(epsilons) > 1:
        epsilon_changes = np.diff(epsilons)
        epsilon_change_rates = epsilon_changes / epsilons[:-1]
        
        metrics["epsilon"] = {
            "initial": epsilons[0],
            "final": epsilons[-1],
            "total_change": epsilons[-1] - epsilons[0],
            "relative_change": (epsilons[-1] - epsilons[0]) / epsilons[0] if epsilons[0] != 0 else 0,
            "avg_change": float(np.mean(epsilon_changes)),
            "avg_change_rate": float(np.mean(epsilon_change_rates))
        }
    
    # 收敛性分析
    if len(node_counts) > 5:
        # 使用最后5次迭代
        last_5_nodes = node_counts[-5:]
        last_5_edges = edge_counts[-5:]
        
        node_growth_rates = np.diff(last_5_nodes) / last_5_nodes[:-1]
        edge_growth_rates = np.diff(last_5_edges) / last_5_edges[:-1]
        
        metrics["convergence"] = {
            "node_growth_rate_mean": float(node_growth_rates.mean()),
            "node_growth_rate_std": float(node_growth_rates.std()),
            "edge_growth_rate_mean": float(edge_growth_rates.mean()),
            "edge_growth_rate_std": float(edge_growth_rates.std()),
            "is_converging": bool(node_growth_rates.mean() < 0.1 and edge_growth_rates.mean() < 0.1)
        }
    
    # 详细指标（如果请求）
    if include_detailed and len(node_counts) > 2:
        # 计算增长率序列
        node_growth_seq = np.diff(node_counts) / node_counts[:-1]
        edge_growth_seq = np.diff(edge_counts) / edge_counts[:-1]
        
        # 自相关性
        if len(node_growth_seq) > 1:
            node_autocorr = np.corrcoef(node_growth_seq[:-1], node_growth_seq[1:])[0, 1]
            edge_autocorr = np.corrcoef(edge_growth_seq[:-1], edge_growth_seq[1:])[0, 1]
        else:
            node_autocorr = edge_autocorr = 0
        
        metrics["detailed"] = {
            "node_growth_sequence": node_growth_seq.tolist(),
            "edge_growth_sequence": edge_growth_seq.tolist(),
            "node_autocorrelation": float(node_autocorr),
            "edge_autocorrelation": float(edge_autocorr),
            "node_edge_correlation": float(np.corrcoef(node_counts, edge_counts)[0, 1]) 
                                    if len(node_counts) > 1 else 0
        }
    
    return metrics

def compare_graphs(graph1: UnitDistanceGraph, 
                  graph2: UnitDistanceGraph,
                  geometry: Optional[GeometryEngine] = None) -> Dict[str, Any]:
    """
    比较两个图
    
    Args:
        graph1: 第一个图
        graph2: 第二个图
        geometry: 几何引擎（用于几何比较）
        
    Returns:
        比较结果字典
    """
    metrics1 = graph_metrics(graph1, geometry)
    metrics2 = graph_metrics(graph2, geometry)
    
    comparison = {
        "size_comparison": {
            "nodes_diff": graph2.num_nodes - graph1.num_nodes,
            "edges_diff": graph2.num_edges - graph1.num_edges,
            "density_diff": graph2.edge_density - graph1.edge_density,
            "nodes_ratio": graph2.num_nodes / graph1.num_nodes if graph1.num_nodes > 0 else float('inf'),
            "edges_ratio": graph2.num_edges / graph1.num_edges if graph1.num_edges > 0 else float('inf')
        },
        "epsilon": {
            "graph1": graph1.epsilon,
            "graph2": graph2.epsilon,
            "diff": graph2.epsilon - graph1.epsilon,
            "relative_diff": (graph2.epsilon - graph1.epsilon) / graph1.epsilon if graph1.epsilon > 0 else 0
        }
    }
    
    # 比较度分布
    if "degree" in metrics1 and "degree" in metrics2:
        comparison["degree"] = {
            "mean_diff": metrics2["degree"]["mean"] - metrics1["degree"]["mean"],
            "mean_ratio": metrics2["degree"]["mean"] / metrics1["degree"]["mean"] 
                          if metrics1["degree"]["mean"] > 0 else float('inf'),
            "max_diff": metrics2["degree"]["max"] - metrics1["degree"]["max"]
        }
    
    # Jaccard相似度（基于边）
    edges1 = set(graph1.edges)
    edges2 = set(graph2.edges)
    
    intersection = len(edges1.intersection(edges2))
    union = len(edges1.union(edges2))
    
    comparison["similarity"] = {
        "jaccard": intersection / union if union > 0 else 0,
        "edges_in_common": intersection,
        "edges_total_union": union,
        "overlap_percentage": intersection / len(edges1) * 100 if edges1 else 0
    }
    
    # 节点位置相似度（如果节点数相同）
    if graph1.num_nodes == graph2.num_nodes:
        # 假设节点顺序相同
        position_diffs = np.linalg.norm(graph1.nodes - graph2.nodes, axis=1)
        comparison["position"] = {
            "mean_distance": float(np.mean(position_diffs)),
            "max_distance": float(np.max(position_diffs)),
            "std_distance": float(np.std(position_diffs))
        }
    
    # 连通性比较
    if "connectivity" in metrics1 and "connectivity" in metrics2:
        comparison["connectivity"] = {
            "both_connected": metrics1["connectivity"]["is_connected"] and 
                             metrics2["connectivity"]["is_connected"],
            "components_diff": metrics2["connectivity"].get("num_components", 1) - 
                              metrics1["connectivity"].get("num_components", 1)
        }
    
    return comparison

def compute_statistical_significance(results: List[Dict[str, Any]], 
                                    metric: str = "chromatic_number",
                                    confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    计算结果的统计显著性
    
    Args:
        results: 结果列表
        metric: 要分析的指标
        confidence_level: 置信水平
        
    Returns:
        统计显著性结果
    """
    if len(results) < 2:
        return {
            "error": "Need at least 2 results for statistical analysis",
            "n": len(results)
        }
    
    # 提取指标值
    values = []
    for result in results:
        if metric in result:
            values.append(result[metric])
        elif "pipeline_result" in result and hasattr(result["pipeline_result"], metric):
            values.append(getattr(result["pipeline_result"], metric))
        elif isinstance(result, dict) and metric in result.get("metrics", {}):
            values.append(result["metrics"][metric])
    
    if len(values) < 2:
        return {
            "error": f"Metric {metric} not found in sufficient results",
            "n_values": len(values)
        }
    
    values = np.array(values)
    
    # 基本统计
    stats_result = {
        "metric": metric,
        "n": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25))
    }
    
    # 正态性检验
    if len(values) >= 3:
        try:
            from scipy.stats import shapiro, normaltest
            # Shapiro-Wilk检验（适合小样本）
            _, shapiro_p = shapiro(values)
            # D'Agostino's K^2检验
            _, dagostino_p = normaltest(values)
            
            stats_result["normality_tests"] = {
                "shapiro_wilk": {
                    "p_value": float(shapiro_p),
                    "is_normal": shapiro_p > 0.05
                },
                "dagostino": {
                    "p_value": float(dagostino_p),
                    "is_normal": dagostino_p > 0.05
                }
            }
        except:
            pass
    
    # 置信区间
    if len(values) >= 2:
        from scipy import stats as sp_stats
        alpha = 1 - confidence_level
        dof = len(values) - 1
        
        t_critical = sp_stats.t.ppf(1 - alpha/2, dof)
        sem = sp_stats.sem(values)
        margin_of_error = t_critical * sem
        
        stats_result["confidence_interval"] = {
            "level": confidence_level,
            "lower": float(values.mean() - margin_of_error),
            "upper": float(values.mean() + margin_of_error),
            "margin_of_error": float(margin_of_error)
        }
    
    # 异常值检测（使用IQR方法）
    q1 = stats_result["q1"]
    q3 = stats_result["q3"]
    iqr = stats_result["iqr"]
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = values[(values < lower_bound) | (values > upper_bound)]
    
    stats_result["outliers"] = {
        "count": len(outliers),
        "values": outliers.tolist(),
        "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
    }
    
    return stats_result

def compute_correlation(data1: List[float], data2: List[float], 
                       method: str = "pearson") -> Dict[str, float]:
    """
    计算两组数据的相关性
    
    Args:
        data1: 第一组数据
        data2: 第二组数据
        method: 相关性方法 ("pearson", "spearman", "kendall")
        
    Returns:
        相关性结果
    """
    if len(data1) != len(data2):
        raise ValueError("Data sets must have the same length")
    
    if len(data1) < 2:
        return {"correlation": 0.0, "p_value": 1.0}
    
    from scipy import stats
    
    if method == "pearson":
        corr, p_value = stats.pearsonr(data1, data2)
    elif method == "spearman":
        corr, p_value = stats.spearmanr(data1, data2)
    elif method == "kendall":
        corr, p_value = stats.kendalltau(data1, data2)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return {
        "correlation": float(corr),
        "p_value": float(p_value),
        "method": method,
        "significant": p_value < 0.05
    }