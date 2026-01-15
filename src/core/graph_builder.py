"""
图构建器模块 - 完整的实现
在基本域内生成点集并构建单位距离图
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from scipy.spatial import KDTree
import networkx as nx

from .geometry_engine import GeometryEngine

@dataclass
class GraphConfig:
    """图构建配置"""
    epsilon: float = 0.02
    sampling_method: str = "fibonacci"  # fibonacci, grid, random, hybrid
    use_kdtree: bool = True
    search_factor: float = 1.5
    min_points: int = 10
    max_points: int = 1000
    jitter: float = 0.0  # 随机扰动
    edge_density_threshold: float = 0.1  # 边密度阈值
    
    def __post_init__(self):
        """验证参数"""
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.min_points <= 0:
            raise ValueError("min_points must be positive")
        if self.max_points < self.min_points:
            raise ValueError("max_points must be >= min_points")

@dataclass
class UnitDistanceGraph:
    """单位距离图数据结构"""
    nodes: np.ndarray  # (N, 2) 节点坐标
    edges: List[Tuple[int, int]]  # 边列表
    epsilon: float  # 距离容差
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证数据"""
        if len(self.nodes.shape) != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must be (N, 2) array")
        
        # 确保边是排序的且没有自环
        self.edges = [(min(u, v), max(u, v)) for u, v in self.edges if u != v]
        self.edges = list(set(self.edges))  # 去重
    
    @property
    def num_nodes(self) -> int:
        """节点数量"""
        return len(self.nodes)
    
    @property
    def num_edges(self) -> int:
        """边数量"""
        return len(self.edges)
    
    @property
    def edge_density(self) -> float:
        """边密度"""
        n = self.num_nodes
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1) / 2
        return self.num_edges / max_edges
    
    def to_networkx(self) -> nx.Graph:
        """转换为NetworkX图"""
        G = nx.Graph()
        
        # 添加节点和位置属性
        for i, pos in enumerate(self.nodes):
            G.add_node(i, pos=pos)
        
        # 添加边
        G.add_edges_from(self.edges)
        
        # 添加元数据
        G.graph.update(self.metadata)
        G.graph['epsilon'] = self.epsilon
        
        return G
    
    def to_adjacency_matrix(self) -> np.ndarray:
        """转换为邻接矩阵"""
        n = self.num_nodes
        adj = np.zeros((n, n), dtype=int)
        
        for u, v in self.edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        return adj
    
    def get_node_degrees(self) -> np.ndarray:
        """获取节点度数"""
        degrees = np.zeros(self.num_nodes, dtype=int)
        
        for u, v in self.edges:
            degrees[u] += 1
            degrees[v] += 1
        
        return degrees
    
    def get_distance_matrix(self, geometry: GeometryEngine) -> np.ndarray:
        """获取距离矩阵"""
        n = self.num_nodes
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = geometry.get_metric(self.nodes[i], self.nodes[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    def add_node(self, new_node: np.ndarray) -> 'UnitDistanceGraph':
        """添加新节点（不添加边）"""
        new_nodes = np.vstack([self.nodes, new_node.reshape(1, 2)])
        return UnitDistanceGraph(
            nodes=new_nodes,
            edges=self.edges.copy(),
            epsilon=self.epsilon,
            metadata=self.metadata.copy()
        )
    
    def add_nodes(self, new_nodes: np.ndarray) -> 'UnitDistanceGraph':
        """添加多个新节点"""
        if len(new_nodes.shape) == 1:
            new_nodes = new_nodes.reshape(1, 2)
        
        new_nodes_array = np.vstack([self.nodes, new_nodes])
        return UnitDistanceGraph(
            nodes=new_nodes_array,
            edges=self.edges.copy(),
            epsilon=self.epsilon,
            metadata=self.metadata.copy()
        )
    
    def copy(self) -> 'UnitDistanceGraph':
        """创建副本"""
        return UnitDistanceGraph(
            nodes=self.nodes.copy(),
            edges=self.edges.copy(),
            epsilon=self.epsilon,
            metadata=self.metadata.copy()
        )

class GraphBuilder:
    """
    图构建器：在基本域内生成点集并构建单位距离图
    """
    
    def __init__(self, geometry: GeometryEngine, config: Optional[GraphConfig] = None):
        """
        初始化图构建器
        
        Args:
            geometry: 几何引擎
            config: 图构建配置
        """
        self.geometry = geometry
        self.config = config or GraphConfig()
        
        # 验证配置
        if self.config.epsilon <= 0:
            raise ValueError("epsilon must be positive")
    
    def initialize_points(self, n_points: int, method: Optional[str] = None, 
                         jitter: Optional[float] = None) -> np.ndarray:
        """
        在基本域内生成点集
        
        Args:
            n_points: 点数
            method: 采样方法（覆盖配置中的方法）
            jitter: 随机扰动幅度
            
        Returns:
            点坐标数组 (n_points, 2)
        """
        if n_points < self.config.min_points:
            raise ValueError(f"Number of points ({n_points}) is less than min_points ({self.config.min_points})")
        
        if n_points > self.config.max_points:
            raise ValueError(f"Number of points ({n_points}) exceeds max_points ({self.config.max_points})")
        
        method = method or self.config.sampling_method
        jitter = jitter or self.config.jitter
        
        if method == "fibonacci":
            nodes = self._initialize_fibonacci_points(n_points, jitter)
        elif method == "grid":
            nodes = self._initialize_grid_points(n_points, jitter)
        elif method == "random":
            nodes = self._initialize_random_points(n_points, jitter)
        elif method == "hybrid":
            nodes = self._initialize_hybrid_points(n_points, jitter)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        return nodes
    
    def _initialize_fibonacci_points(self, n_points: int, jitter: float) -> np.ndarray:
        """使用Fibonacci晶格生成点"""
        phi = (1 + np.sqrt(5)) / 2  # 黄金比例
        points = []
        
        for i in range(1, n_points + 1):
            # Fibonacci晶格
            u = (i / phi) % 1
            v = i / n_points
            
            # 转换为欧几里得坐标
            lattice_coords = np.array([u, v])
            point = self.geometry.to_euclidean_coords(lattice_coords)
            
            # 添加随机扰动
            if jitter > 0:
                point += np.random.uniform(-jitter, jitter, 2)
            
            points.append(point)
        
        return np.array(points)
    
    def _initialize_grid_points(self, n_points: int, jitter: float) -> np.ndarray:
        """使用网格生成点"""
        # 计算网格大小
        n_per_axis = int(np.ceil(np.sqrt(n_points)))
        actual_points = n_per_axis ** 2
        
        points = []
        
        for i in range(n_per_axis):
            for j in range(n_per_axis):
                # 均匀网格
                u = (i + 0.5) / n_per_axis
                v = (j + 0.5) / n_per_axis
                
                # 转换为欧几里得坐标
                lattice_coords = np.array([u, v])
                point = self.geometry.to_euclidean_coords(lattice_coords)
                
                # 添加随机扰动
                if jitter > 0:
                    point += np.random.uniform(-jitter, jitter, 2)
                
                points.append(point)
        
        # 如果生成的点数多于需要的点数，随机选择
        if actual_points > n_points:
            indices = np.random.choice(actual_points, n_points, replace=False)
            points = [points[i] for i in indices]
        
        return np.array(points)
    
    def _initialize_random_points(self, n_points: int, jitter: float) -> np.ndarray:
        """生成随机点"""
        points = []
        
        for _ in range(n_points):
            # 在基本域内随机采样
            u, v = np.random.uniform(0, 1, 2)
            
            # 转换为欧几里得坐标
            lattice_coords = np.array([u, v])
            point = self.geometry.to_euclidean_coords(lattice_coords)
            
            # 添加随机扰动（对于随机点，jitter已经有意义）
            if jitter > 0:
                point += np.random.uniform(-jitter, jitter, 2)
            
            points.append(point)
        
        return np.array(points)
    
    def _initialize_hybrid_points(self, n_points: int, jitter: float) -> np.ndarray:
        """混合采样方法"""
        # 使用Fibonacci生成大部分点
        fib_points = self._initialize_fibonacci_points(n_points, jitter=0)
        
        # 添加一些随机点
        n_random = max(1, n_points // 10)
        random_points = self._initialize_random_points(n_random, jitter)
        
        # 组合
        all_points = np.vstack([fib_points, random_points])
        
        # 如果点数超过需求，随机选择
        if len(all_points) > n_points:
            indices = np.random.choice(len(all_points), n_points, replace=False)
            all_points = all_points[indices]
        
        return all_points
    
    def build_edges_naive(self, nodes: np.ndarray) -> List[Tuple[int, int]]:
        """
        朴素方法构建边：检查所有点对
        
        Args:
            nodes: 节点坐标数组 (N, 2)
            
        Returns:
            边列表 [(u, v), ...]
        """
        n = len(nodes)
        edges = []
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.geometry.get_metric(nodes[i], nodes[j])
                if abs(dist - 1.0) < self.config.epsilon:
                    edges.append((i, j))
        
        return edges
    
    def build_edges_kdtree(self, nodes: np.ndarray, 
                          search_factor: Optional[float] = None) -> List[Tuple[int, int]]:
        """
        使用KDTree加速构建边
        
        Args:
            nodes: 节点坐标数组 (N, 2)
            search_factor: 搜索半径因子
            
        Returns:
            边列表 [(u, v), ...]
        """
        search_factor = search_factor or self.config.search_factor
        
        # 使用KDTree进行最近邻搜索
        tree = KDTree(nodes)
        
        # 在商空间上搜索半径为 (1+epsilon)*search_factor 内的点
        search_radius = (1.0 + self.config.epsilon) * search_factor
        
        # 查询所有在搜索半径内的点对
        pairs = tree.query_pairs(search_radius)
        
        # 检查每个点对的实际距离
        edges = []
        for i, j in pairs:
            dist = self.geometry.get_metric(nodes[i], nodes[j])
            if abs(dist - 1.0) < self.config.epsilon:
                edges.append((i, j))
        
        return edges
    
    def construct_graph(self, nodes: np.ndarray, 
                       use_kdtree: Optional[bool] = None) -> UnitDistanceGraph:
        """
        构建单位距离图
        
        Args:
            nodes: 节点坐标
            use_kdtree: 是否使用KDTree加速
            
        Returns:
            UnitDistanceGraph对象
        """
        use_kdtree = use_kdtree if use_kdtree is not None else self.config.use_kdtree
        
        if use_kdtree:
            edges = self.build_edges_kdtree(nodes)
        else:
            edges = self.build_edges_naive(nodes)
        
        # 创建元数据
        metadata = {
            "sampling_method": self.config.sampling_method,
            "epsilon": self.config.epsilon,
            "construction_method": "kdtree" if use_kdtree else "naive",
            "num_nodes": len(nodes),
            "num_edges": len(edges)
        }
        
        return UnitDistanceGraph(
            nodes=nodes,
            edges=edges,
            epsilon=self.config.epsilon,
            metadata=metadata
        )
    
    def add_node_with_edges(self, graph: UnitDistanceGraph, 
                           new_node: np.ndarray) -> UnitDistanceGraph:
        """
        向图中添加新节点并构建边
        
        Args:
            graph: 现有图
            new_node: 新节点坐标
            
        Returns:
            更新后的图
        """
        # 添加新节点
        new_nodes = np.vstack([graph.nodes, new_node.reshape(1, 2)])
        new_node_idx = len(graph.nodes)
        
        # 检查新节点与所有现有节点的距离
        new_edges = []
        for i, node in enumerate(graph.nodes):
            dist = self.geometry.get_metric(new_node, node)
            if abs(dist - 1.0) < graph.epsilon:
                new_edges.append((new_node_idx, i))
        
        # 更新边列表
        updated_edges = graph.edges + new_edges
        
        # 更新元数据
        updated_metadata = graph.metadata.copy()
        updated_metadata["num_nodes"] = len(new_nodes)
        updated_metadata["num_edges"] = len(updated_edges)
        
        return UnitDistanceGraph(
            nodes=new_nodes,
            edges=updated_edges,
            epsilon=graph.epsilon,
            metadata=updated_metadata
        )
    
    def add_nodes_with_edges(self, graph: UnitDistanceGraph,
                            new_nodes: np.ndarray) -> UnitDistanceGraph:
        """
        向图中添加多个新节点并构建边
        
        Args:
            graph: 现有图
            new_nodes: 新节点坐标数组
            
        Returns:
            更新后的图
        """
        if len(new_nodes.shape) == 1:
            new_nodes = new_nodes.reshape(1, 2)
        
        current_graph = graph.copy()
        
        # 逐个添加节点
        for new_node in new_nodes:
            current_graph = self.add_node_with_edges(current_graph, new_node)
        
        return current_graph
    
    def filter_sparse_nodes(self, graph: UnitDistanceGraph, 
                           min_degree: int = 1) -> UnitDistanceGraph:
        """
        过滤度数过低的节点
        
        Args:
            graph: 输入图
            min_degree: 最小度数
            
        Returns:
            过滤后的图
        """
        # 计算节点度数
        degrees = graph.get_node_degrees()
        
        # 找到度数足够的节点
        valid_indices = np.where(degrees >= min_degree)[0]
        
        if len(valid_indices) == 0:
            # 如果没有节点满足条件，返回空图
            return UnitDistanceGraph(
                nodes=np.zeros((0, 2)),
                edges=[],
                epsilon=graph.epsilon,
                metadata=graph.metadata.copy()
            )
        
        # 创建节点映射
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
        
        # 提取有效节点
        filtered_nodes = graph.nodes[valid_indices]
        
        # 提取有效边
        filtered_edges = []
        for u, v in graph.edges:
            if u in valid_indices and v in valid_indices:
                new_u = index_map[u]
                new_v = index_map[v]
                filtered_edges.append((new_u, new_v))
        
        # 更新元数据
        updated_metadata = graph.metadata.copy()
        updated_metadata["num_nodes"] = len(filtered_nodes)
        updated_metadata["num_edges"] = len(filtered_edges)
        updated_metadata["filtered"] = True
        updated_metadata["min_degree"] = min_degree
        
        return UnitDistanceGraph(
            nodes=filtered_nodes,
            edges=filtered_edges,
            epsilon=graph.epsilon,
            metadata=updated_metadata
        )
    
    def merge_graphs(self, graph1: UnitDistanceGraph, 
                    graph2: UnitDistanceGraph) -> UnitDistanceGraph:
        """
        合并两个图
        
        Args:
            graph1: 第一个图
            graph2: 第二个图
            
        Returns:
            合并后的图
        """
        if abs(graph1.epsilon - graph2.epsilon) > 1e-10:
            raise ValueError("Graphs must have the same epsilon for merging")
        
        # 合并节点
        merged_nodes = np.vstack([graph1.nodes, graph2.nodes])
        
        # 合并边（调整第二个图的节点索引）
        offset = len(graph1.nodes)
        adjusted_edges2 = [(u + offset, v + offset) for u, v in graph2.edges]
        merged_edges = graph1.edges + adjusted_edges2
        
        # 检查节点间的边（不同图之间的节点）
        for i, node1 in enumerate(graph1.nodes):
            for j, node2 in enumerate(graph2.nodes):
                dist = self.geometry.get_metric(node1, node2)
                if abs(dist - 1.0) < graph1.epsilon:
                    merged_edges.append((i, j + offset))
        
        # 合并元数据
        merged_metadata = {
            **graph1.metadata,
            **graph2.metadata,
            "merged": True,
            "graph1_nodes": len(graph1.nodes),
            "graph2_nodes": len(graph2.nodes),
            "total_nodes": len(merged_nodes),
            "total_edges": len(merged_edges)
        }
        
        return UnitDistanceGraph(
            nodes=merged_nodes,
            edges=merged_edges,
            epsilon=graph1.epsilon,
            metadata=merged_metadata
        )
    
    def save_graph(self, graph: UnitDistanceGraph, filepath: str):
        """
        保存图到文件
        
        Args:
            graph: 要保存的图
            filepath: 文件路径
        """
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'nodes': graph.nodes,
                'edges': graph.edges,
                'epsilon': graph.epsilon,
                'metadata': graph.metadata
            }, f)
    
    @classmethod
    def load_graph(cls, filepath: str) -> UnitDistanceGraph:
        """
        从文件加载图
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的图
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return UnitDistanceGraph(
            nodes=data['nodes'],
            edges=data['edges'],
            epsilon=data['epsilon'],
            metadata=data.get('metadata', {})
        )