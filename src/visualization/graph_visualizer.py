"""
图可视化器 - 完整实现
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
from typing import Dict, Any, Optional, List, Tuple
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..core.graph_builder import UnitDistanceGraph
from ..core.geometry_engine import GeometryEngine

class GraphVisualizer:
    """图可视化器"""
    
    def __init__(self, style: str = "default"):
        """
        初始化可视化器
        
        Args:
            style: 绘图风格 ("default", "dark", "minimal")
        """
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """设置绘图风格"""
        if self.style == "dark":
            plt.style.use('dark_background')
            self.node_color = 'white'
            self.edge_color = 'gray'
            self.domain_color = 'red'
            self.text_color = 'white'
        elif self.style == "minimal":
            self.node_color = 'black'
            self.edge_color = '#666666'
            self.domain_color = 'red'
            self.text_color = 'black'
        else:  # default
            self.node_color = 'blue'
            self.edge_color = 'black'
            self.domain_color = 'red'
            self.text_color = 'black'
    
    def plot_graph(self, graph: UnitDistanceGraph, 
                  geometry: GeometryEngine,
                  coloring: Optional[Dict[int, int]] = None,
                  title: str = "Unit Distance Graph",
                  figsize: Tuple[int, int] = (10, 8),
                  show_labels: bool = False,
                  node_size: int = 50,
                  edge_alpha: float = 0.5,
                  show_domain: bool = True) -> plt.Figure:
        """
        绘制单位距离图
        
        Args:
            graph: 单位距离图
            geometry: 几何引擎
            coloring: 染色方案（可选）
            title: 图像标题
            figsize: 图形大小
            show_labels: 是否显示节点标签
            node_size: 节点大小
            edge_alpha: 边透明度
            show_domain: 是否显示基本域
            
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制基本域
        if show_domain:
            vertices = geometry.get_fundamental_domain_vertices()
            domain = np.vstack([vertices, vertices[0]])  # 闭合多边形
            ax.plot(domain[:, 0], domain[:, 1], '--', 
                   color=self.domain_color, alpha=0.7, linewidth=1.5,
                   label='Fundamental Domain')
        
        # 绘制边
        if graph.edges:
            edges = []
            for u, v in graph.edges:
                edges.append([graph.nodes[u], graph.nodes[v]])
            
            lc = LineCollection(edges, colors=self.edge_color, 
                               alpha=edge_alpha, linewidths=0.8)
            ax.add_collection(lc)
        
        # 绘制节点
        if coloring is not None:
            # 使用染色方案
            colors = plt.cm.tab10.colors
            for i, node in enumerate(graph.nodes):
                color_idx = coloring.get(i, 0) % len(colors)
                ax.scatter(node[0], node[1], color=colors[color_idx], 
                          s=node_size, edgecolors='black', zorder=5)
        else:
            ax.scatter(graph.nodes[:, 0], graph.nodes[:, 1], 
                      color=self.node_color, s=node_size, 
                      edgecolors='black', zorder=5)
        
        # 显示节点标签
        if show_labels:
            for i, node in enumerate(graph.nodes):
                ax.annotate(str(i), node, fontsize=8, 
                           color=self.text_color, ha='center', va='center')
        
        # 设置图形属性
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axis('equal')
        
        if show_domain:
            ax.legend(loc='upper right')
        
        # 添加图信息文本
        info_text = f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}, ε: {graph.epsilon:.4f}"
        if coloring is not None:
            colors_used = len(set(coloring.values()))
            info_text += f", Colors: {colors_used}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_cover(self, graph: UnitDistanceGraph,
                  geometry: GeometryEngine,
                  num_copies: Tuple[int, int] = (3, 3),
                  title: str = "Cover Space Visualization",
                  figsize: Tuple[int, int] = (12, 10),
                  node_size: int = 30,
                  edge_alpha: float = 0.3) -> plt.Figure:
        """
        在覆盖空间中绘制图
        
        Args:
            graph: 单位距离图
            geometry: 几何引擎
            num_copies: 在u和v方向上显示的副本数量
            title: 图像标题
            figsize: 图形大小
            node_size: 节点大小
            edge_alpha: 边透明度
            
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        nu, nv = num_copies
        all_edges = []
        
        # 绘制多个基本域副本
        for i in range(-nu//2, nu//2 + 1):
            for j in range(-nv//2, nv//2 + 1):
                shift = geometry.to_euclidean_coords(np.array([i, j]))
                
                # 绘制基本域边界
                vertices = geometry.get_fundamental_domain_vertices()
                shifted_vertices = vertices + shift.reshape(1, 2)
                domain = np.vstack([shifted_vertices, shifted_vertices[0]])
                ax.plot(domain[:, 0], domain[:, 1], 'k--', 
                       alpha=0.2, linewidth=0.5)
        
        # 收集所有边
        for u, v in graph.edges:
            node_u = graph.nodes[u]
            node_v = graph.nodes[v]
            
            # 获取两个节点的副本
            copies_u = geometry.lift_to_cover(node_u, num_copies)
            copies_v = geometry.lift_to_cover(node_v, num_copies)
            
            # 连接距离为1的副本
            for copy_u in copies_u:
                for copy_v in copies_v:
                    dist = np.linalg.norm(copy_u - copy_v)
                    if abs(dist - 1.0) < 1e-3:  # 严格的距离检查
                        all_edges.append([copy_u, copy_v])
        
        # 绘制所有边
        if all_edges:
            lc = LineCollection(all_edges, colors='black', 
                               alpha=edge_alpha, linewidths=0.5)
            ax.add_collection(lc)
        
        # 绘制所有节点副本
        all_nodes = []
        for node in graph.nodes:
            copies = geometry.lift_to_cover(node, num_copies)
            all_nodes.extend(copies)
        
        if all_nodes:
            all_nodes = np.array(all_nodes)
            ax.scatter(all_nodes[:, 0], all_nodes[:, 1], 
                      color=self.node_color, s=node_size, 
                      edgecolors='black', zorder=5, alpha=0.7)
        
        # 设置图形属性
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.axis('equal')
        
        # 添加图信息文本
        info_text = (f"Original Graph: {graph.num_nodes} nodes, {graph.num_edges} edges\n"
                    f"Cover: {num_copies[0]}×{num_copies[1]} copies")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_degree_distribution(self, graph: UnitDistanceGraph,
                                figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        绘制度分布
        
        Args:
            graph: 单位距离图
            figsize: 图形大小
            
        Returns:
            matplotlib图形对象
        """
        import networkx as nx
        
        nx_graph = graph.to_networkx()
        degrees = list(dict(nx_graph.degree()).values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 直方图
        ax1.hist(degrees, bins=min(20, max(degrees)+1), 
                edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Degree')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Degree Histogram')
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        ax2.boxplot(degrees, vert=False)
        ax2.set_xlabel('Degree')
        ax2.set_title('Degree Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = (f"Mean: {np.mean(degrees):.2f}\n"
                     f"Std: {np.std(degrees):.2f}\n"
                     f"Min: {min(degrees)}\n"
                     f"Max: {max(degrees)}")
        
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Degree Distribution (Nodes: {graph.num_nodes}, Edges: {graph.num_edges})')
        plt.tight_layout()
        
        return fig
    
    def plot_coloring_analysis(self, graph: UnitDistanceGraph,
                              coloring: Dict[int, int],
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制染色分析
        
        Args:
            graph: 单位距离图
            coloring: 染色方案
            figsize: 图形大小
            
        Returns:
            matplotlib图形对象
        """
        if not coloring:
            raise ValueError("Empty coloring")
        
        # 计算颜色统计
        color_values = list(coloring.values())
        unique_colors = set(color_values)
        color_counts = {color: color_values.count(color) for color in unique_colors}
        
        # 计算每个节点的冲突
        conflicts_per_node = []
        for u, v in graph.edges:
            if u in coloring and v in coloring:
                if coloring[u] == coloring[v]:
                    conflicts_per_node.append(u)
                    conflicts_per_node.append(v)
        
        fig = plt.figure(figsize=figsize)
        
        # 1. 颜色分布饼图
        ax1 = plt.subplot(2, 2, 1)
        colors = plt.cm.tab10.colors
        color_labels = [f'Color {c}' for c in sorted(unique_colors)]
        color_sizes = [color_counts[c] for c in sorted(unique_colors)]
        
        wedges, texts, autotexts = ax1.pie(color_sizes, labels=color_labels, 
                                          autopct='%1.1f%%', startangle=90)
        
        # 设置颜色
        for i, wedge in enumerate(wedges):
            color_idx = list(sorted(unique_colors))[i] % len(colors)
            wedge.set_facecolor(colors[color_idx])
        
        ax1.set_title('Color Distribution')
        
        # 2. 颜色数量条形图
        ax2 = plt.subplot(2, 2, 2)
        color_indices = list(sorted(unique_colors))
        counts = [color_counts[c] for c in color_indices]
        
        bars = ax2.bar(range(len(color_indices)), counts)
        
        # 设置颜色
        for i, bar in enumerate(bars):
            color_idx = color_indices[i] % len(colors)
            bar.set_color(colors[color_idx])
        
        ax2.set_xlabel('Color Index')
        ax2.set_ylabel('Count')
        ax2.set_title('Color Counts')
        ax2.set_xticks(range(len(color_indices)))
        ax2.set_xticklabels(color_indices)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 冲突分析
        ax3 = plt.subplot(2, 2, 3)
        
        if conflicts_per_node:
            conflict_counts = {}
            for node in conflicts_per_node:
                conflict_counts[node] = conflict_counts.get(node, 0) + 1
            
            nodes = list(conflict_counts.keys())
            counts = list(conflict_counts.values())
            
            ax3.bar(range(len(nodes)), counts)
            ax3.set_xlabel('Node Index')
            ax3.set_ylabel('Conflict Count')
            ax3.set_title(f'Node Conflicts (Total: {len(set(conflicts_per_node))})')
            ax3.set_xticks(range(len(nodes)))
            ax3.set_xticklabels(nodes)
        else:
            ax3.text(0.5, 0.5, 'No Conflicts', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Node Conflicts')
        
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 统计信息
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        stats_text = (f"Total Nodes: {graph.num_nodes}\n"
                     f"Colored Nodes: {len(coloring)}\n"
                     f"Colors Used: {len(unique_colors)}\n"
                     f"Total Conflicts: {len(set(conflicts_per_node))}\n"
                     f"Balancing Entropy: {self._compute_entropy(list(color_counts.values())):.3f}")
        
        ax4.text(0.1, 0.5, stats_text, fontsize=12, 
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Coloring Analysis', fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def _compute_entropy(self, values):
        """计算分布的熵"""
        total = sum(values)
        if total == 0:
            return 0.0
        
        probabilities = [v / total for v in values]
        entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
        return entropy
    
    def plot_pipeline_history(self, history: List[Dict[str, Any]],
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制管道运行历史
        
        Args:
            history: 管道历史
            figsize: 图形大小
            
        Returns:
            matplotlib图形对象
        """
        if not history:
            raise ValueError("Empty history")
        
        # 提取数据
        iterations = []
        node_counts = []
        edge_counts = []
        epsilons = []
        
        for entry in history:
            if "nodes" in entry and "edges" in entry:
                iterations.append(entry.get("iteration", 0))
                node_counts.append(entry["nodes"])
                edge_counts.append(entry["edges"])
                if "epsilon" in entry:
                    epsilons.append(entry["epsilon"])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 节点和边数量增长
        ax1.plot(iterations, node_counts, 'b-', label='Nodes', linewidth=2)
        ax1.plot(iterations, edge_counts, 'r-', label='Edges', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Count')
        ax1.set_title('Graph Growth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 边密度
        densities = []
        for n, e in zip(node_counts, edge_counts):
            if n > 1:
                densities.append(e / (n * (n - 1) / 2))
            else:
                densities.append(0)
        
        ax2.plot(iterations, densities, 'g-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Edge Density')
        ax2.set_title('Edge Density Over Time')
        ax2.grid(True, alpha=0.3)
        
        # 3. Epsilon变化
        if epsilons:
            ax3.plot(iterations[:len(epsilons)], epsilons, 'm-', linewidth=2)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Epsilon')
            ax3.set_title('Epsilon Annealing')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Epsilon Data', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Epsilon Annealing')
        
        # 4. 增长率
        if len(node_counts) > 1:
            node_growth = np.diff(node_counts)
            edge_growth = np.diff(edge_counts)
            
            ax4.plot(iterations[1:], node_growth, 'b-', label='Node Growth', linewidth=2)
            ax4.plot(iterations[1:], edge_growth, 'r-', label='Edge Growth', linewidth=2)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Growth per Iteration')
            ax4.set_title('Growth Rates')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Growth Rates')
        
        plt.suptitle('Pipeline History Analysis', fontsize=16)
        plt.tight_layout()
        
        return fig