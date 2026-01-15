"""
优化器模块 - 完整的实现
智能增强与梯度优化，通过反馈循环寻找难以染色的点
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import warnings

from .geometry_engine import GeometryEngine
from .graph_builder import GraphBuilder, UnitDistanceGraph
from .sat_solver import SATSolver

@dataclass
class OptimizerConfig:
    """优化器配置"""
    method: str = "constraint_based"  # energy_based, constraint_based, hybrid, evolutionary
    num_candidates: int = 1000
    relaxation_iterations: int = 5
    learning_rate: float = 0.1
    energy_sigma: float = 0.02
    attraction_factor: float = 0.1
    max_energy_evaluations: int = 1000
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    
    def __post_init__(self):
        """验证参数"""
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.energy_sigma <= 0:
            raise ValueError("energy_sigma must be positive")

class GraphOptimizer:
    """
    图优化器：通过反馈循环增强图，寻找难以染色的点
    """
    
    def __init__(self, geometry: GeometryEngine, builder: GraphBuilder,
                 config: Optional[OptimizerConfig] = None):
        """
        初始化优化器
        
        Args:
            geometry: 几何引擎
            builder: 图构建器
            config: 优化器配置
        """
        self.geometry = geometry
        self.builder = builder
        self.config = config or OptimizerConfig()
        
        # 优化历史
        self.history = []
    
    def find_hard_point(self, graph: UnitDistanceGraph,
                       coloring: Dict[int, int],
                       method: Optional[str] = None) -> np.ndarray:
        """
        寻找难以染色的点
        
        Args:
            graph: 当前图
            coloring: 当前染色方案
            method: 寻找方法（覆盖配置中的方法）
            
        Returns:
            新点的坐标
        """
        method = method or self.config.method
        
        if method == "energy_based":
            return self.find_hard_point_energy_based(graph, coloring)
        elif method == "constraint_based":
            return self.find_hard_point_constraint_based(graph, coloring)
        elif method == "hybrid":
            return self.find_hard_point_hybrid(graph, coloring)
        elif method == "evolutionary":
            return self.find_hard_point_evolutionary(graph, coloring)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def find_hard_point_energy_based(self, graph: UnitDistanceGraph,
                                    coloring: Dict[int, int]) -> np.ndarray:
        """
        基于能量函数寻找难以染色的点
        
        Args:
            graph: 当前图
            coloring: 当前染色方案
            
        Returns:
            新点的坐标
        """
        nodes = graph.nodes
        n_nodes = len(nodes)
        epsilon = graph.epsilon
        
        # 如果没有染色方案，返回随机点
        if not coloring:
            return self._random_point_in_fundamental_domain()
        
        # 定义能量函数
        def energy_function(x):
            """能量函数：我们希望最小化能量（即最大化冲突）"""
            point = np.array(x)
            energy = 0.0
            
            for i, node in enumerate(nodes):
                dist = self.geometry.get_metric(point, node)
                
                # 使用高斯核计算能量贡献
                if i in coloring:
                    color = coloring[i]
                    # 权重：我们希望与不同颜色的节点距离为1
                    # 这里使用颜色相关的权重
                    weight = 1.0  # 可以根据颜色调整权重
                    
                    # 高斯能量：距离为1时能量最小
                    gaussian = np.exp(-((dist - 1.0) ** 2) / (2 * self.config.energy_sigma ** 2))
                    energy -= weight * gaussian
                
                # 添加排斥项：避免与同色节点距离为1
                # 这可以在高阶优化中实现
            
            return energy
        
        # 多次随机起始点以寻找全局最小值
        best_point = None
        best_energy = float('inf')
        
        n_restarts = min(10, self.config.num_candidates // 100)
        
        for restart in range(n_restarts):
            # 随机起始点
            x0 = self._random_point_in_fundamental_domain()
            
            try:
                # 使用局部优化
                result = minimize(
                    energy_function,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(None, None), (None, None)],  # 无界，因为几何引擎会处理周期性
                    options={
                        'maxiter': 100,
                        'disp': False
                    }
                )
                
                if result.success and result.fun < best_energy:
                    best_energy = result.fun
                    best_point = result.x
                    
            except Exception as e:
                if self.config.verbose:
                    warnings.warn(f"Optimization failed: {e}")
        
        # 如果优化失败，使用最佳候选点
        if best_point is None:
            # 在候选点中搜索
            candidates = self._generate_candidate_points(self.config.num_candidates)
            best_candidate = None
            best_candidate_energy = float('inf')
            
            for candidate in candidates:
                energy = energy_function(candidate)
                if energy < best_candidate_energy:
                    best_candidate_energy = energy
                    best_candidate = candidate
            
            best_point = best_candidate
        
        # 包裹到基本域
        wrapped_point, _ = self.geometry.wrap_to_fundamental_domain(best_point)
        
        # 记录到历史
        self.history.append({
            "method": "energy_based",
            "point": wrapped_point.copy(),
            "energy": best_energy if best_point is not None else float('inf')
        })
        
        return wrapped_point
    
    def find_hard_point_constraint_based(self, graph: UnitDistanceGraph,
                                        coloring: Dict[int, int]) -> np.ndarray:
        """
        基于约束寻找难以染色的点
        
        策略：寻找一个点，使得它与尽可能多的不同颜色的节点距离为1
        
        Args:
            graph: 当前图
            coloring: 当前染色方案
            
        Returns:
            新点的坐标
        """
        nodes = graph.nodes
        n_nodes = len(nodes)
        epsilon = graph.epsilon
        
        # 如果没有染色方案，返回随机点
        if not coloring:
            return self._random_point_in_fundamental_domain()
        
        # 生成候选点
        candidates = self._generate_candidate_points(self.config.num_candidates)
        
        best_point = None
        best_score = -1
        
        for candidate in candidates:
            # 计算分数
            score = 0
            color_neighbors = set()  # 收集邻居的颜色
            
            for i, node in enumerate(nodes):
                if i not in coloring:
                    continue
                    
                dist = self.geometry.get_metric(candidate, node)
                
                if abs(dist - 1.0) < epsilon:
                    color = coloring[i]
                    color_neighbors.add(color)
            
            # 分数 = 不同颜色的邻居数量
            score = len(color_neighbors)
            
            # 额外的奖励：如果与多个同色节点距离为1，会制造冲突
            # 这里我们计算颜色分布的熵
            if score > 0:
                # 计算每个颜色的邻居数量
                color_counts = {}
                for i, node in enumerate(nodes):
                    if i not in coloring:
                        continue
                        
                    dist = self.geometry.get_metric(candidate, node)
                    if abs(dist - 1.0) < epsilon:
                        color = coloring[i]
                        color_counts[color] = color_counts.get(color, 0) + 1
                
                # 计算熵（鼓励颜色分布均匀）
                total = sum(color_counts.values())
                entropy = 0.0
                for count in color_counts.values():
                    p = count / total
                    entropy -= p * np.log(p + 1e-10)
                
                # 将熵纳入分数
                score += 0.5 * entropy
            
            if score > best_score:
                best_score = score
                best_point = candidate
        
        # 记录到历史
        self.history.append({
            "method": "constraint_based",
            "point": best_point.copy() if best_point is not None else None,
            "score": best_score
        })
        
        return best_point if best_point is not None else self._random_point_in_fundamental_domain()
    
    def find_hard_point_hybrid(self, graph: UnitDistanceGraph,
                              coloring: Dict[int, int]) -> np.ndarray:
        """
        混合方法：先使用约束方法找到候选点，再用能量方法优化
        
        Args:
            graph: 当前图
            coloring: 当前染色方案
            
        Returns:
            新点的坐标
        """
        # 先用约束方法找到好的起始点
        constraint_point = self.find_hard_point_constraint_based(graph, coloring)
        
        # 如果没有有效的染色方案，直接返回
        if not coloring:
            return constraint_point
        
        # 然后用能量方法优化
        nodes = graph.nodes
        epsilon = graph.epsilon
        
        # 定义能量函数（与energy_based方法相同）
        def energy_function(x):
            point = np.array(x)
            energy = 0.0
            
            for i, node in enumerate(nodes):
                dist = self.geometry.get_metric(point, node)
                
                if i in coloring:
                    color = coloring[i]
                    weight = 1.0
                    
                    # 高斯能量：距离为1时能量最小
                    gaussian = np.exp(-((dist - 1.0) ** 2) / (2 * self.config.energy_sigma ** 2))
                    energy -= weight * gaussian
            
            return energy
        
        # 从约束点开始优化
        try:
            result = minimize(
                energy_function,
                constraint_point,
                method='L-BFGS-B',
                bounds=[(None, None), (None, None)],
                options={
                    'maxiter': 50,
                    'disp': False
                }
            )
            
            if result.success:
                optimized_point = result.x
            else:
                optimized_point = constraint_point
                
        except Exception:
            optimized_point = constraint_point
        
        # 包裹到基本域
        wrapped_point, _ = self.geometry.wrap_to_fundamental_domain(optimized_point)
        
        # 记录到历史
        self.history.append({
            "method": "hybrid",
            "constraint_point": constraint_point.copy(),
            "optimized_point": wrapped_point.copy(),
            "energy": energy_function(wrapped_point) if coloring else 0.0
        })
        
        return wrapped_point
    
    def find_hard_point_evolutionary(self, graph: UnitDistanceGraph,
                                    coloring: Dict[int, int]) -> np.ndarray:
        """
        使用进化算法寻找难以染色的点
        
        Args:
            graph: 当前图
            coloring: 当前染色方案
            
        Returns:
            新点的坐标
        """
        if not coloring:
            return self._random_point_in_fundamental_domain()
        
        nodes = graph.nodes
        epsilon = graph.epsilon
        
        # 定义适应度函数（我们想最大化冲突）
        def fitness_function(x):
            point = np.array(x)
            fitness = 0.0
            
            # 收集不同颜色的邻居
            color_neighbors = set()
            
            for i, node in enumerate(nodes):
                if i not in coloring:
                    continue
                    
                dist = self.geometry.get_metric(point, node)
                
                if abs(dist - 1.0) < epsilon:
                    color = coloring[i]
                    color_neighbors.add(color)
            
            # 基本分数：不同颜色的邻居数量
            fitness = len(color_neighbors)
            
            # 如果有点距离为1，进一步计算细节
            if fitness > 0:
                # 计算每个颜色的邻居数量
                color_counts = {}
                for i, node in enumerate(nodes):
                    if i not in coloring:
                        continue
                        
                    dist = self.geometry.get_metric(point, node)
                    if abs(dist - 1.0) < epsilon:
                        color = coloring[i]
                        color_counts[color] = color_counts.get(color, 0) + 1
                
                # 鼓励均匀分布（高熵）
                total = sum(color_counts.values())
                entropy = 0.0
                for count in color_counts.values():
                    p = count / total
                    entropy -= p * np.log(p + 1e-10)
                
                fitness += 0.5 * entropy
            
            return fitness
        
        # 定义搜索边界（在晶格坐标中）
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        
        # 使用差分进化算法
        try:
            result = differential_evolution(
                lambda x: -fitness_function(self.geometry.to_euclidean_coords(x)),  # 最小化负适应度
                bounds,
                maxiter=50,
                popsize=self.config.population_size,
                mutation=self.config.mutation_rate,
                recombination=self.config.crossover_rate,
                disp=False,
                seed=42
            )
            
            if result.success:
                # 转换回欧几里得坐标
                best_lattice_coords = result.x
                best_point = self.geometry.to_euclidean_coords(best_lattice_coords)
            else:
                best_point = self._random_point_in_fundamental_domain()
                
        except Exception as e:
            if self.config.verbose:
                warnings.warn(f"Evolutionary algorithm failed: {e}")
            best_point = self._random_point_in_fundamental_domain()
        
        # 记录到历史
        self.history.append({
            "method": "evolutionary",
            "point": best_point.copy(),
            "fitness": fitness_function(best_point) if best_point is not None else 0.0
        })
        
        return best_point
    
    def _random_point_in_fundamental_domain(self) -> np.ndarray:
        """在基本域内生成随机点"""
        u, v = np.random.uniform(0, 1, 2)
        lattice_coords = np.array([u, v])
        return self.geometry.to_euclidean_coords(lattice_coords)
    
    def _generate_candidate_points(self, n_points: int) -> List[np.ndarray]:
        """生成候选点"""
        candidates = []
        
        # 策略1：均匀随机点
        n_random = n_points // 2
        for _ in range(n_random):
            candidates.append(self._random_point_in_fundamental_domain())
        
        # 策略2：在现有节点周围生成点
        # （这部分需要图信息，所以在这里不实现）
        # 剩余的用随机点填充
        for _ in range(n_points - n_random):
            candidates.append(self._random_point_in_fundamental_domain())
        
        return candidates
    
    def relax_nodes(self, graph: UnitDistanceGraph, 
                   iterations: Optional[int] = None,
                   learning_rate: Optional[float] = None) -> UnitDistanceGraph:
        """
        松弛节点位置以更好地满足单位距离约束
        
        Args:
            graph: 当前图
            iterations: 迭代次数
            learning_rate: 学习率
            
        Returns:
            更新后的图
        """
        iterations = iterations or self.config.relaxation_iterations
        learning_rate = learning_rate or self.config.learning_rate
        
        nodes = graph.nodes.copy()
        edges = graph.edges
        epsilon = graph.epsilon
        n_nodes = len(nodes)
        
        # 如果没有边，直接返回
        if len(edges) == 0:
            return graph.copy()
        
        # 多次迭代松弛
        for iteration in range(iterations):
            gradients = np.zeros_like(nodes)
            
            # 计算每个节点的梯度
            for u, v in edges:
                # 计算当前距离
                dist = self.geometry.get_metric(nodes[u], nodes[v])
                
                # 计算误差
                error = dist - 1.0
                
                if abs(error) > 1e-10:  # 避免除以零
                    # 计算方向向量（考虑周期性）
                    delta = nodes[v] - nodes[u]
                    
                    # 转换为晶格坐标的差值
                    delta_lattice = self.geometry.to_lattice_coords(delta)
                    
                    # 找到最近的整数平移
                    nearest_int = np.round(delta_lattice)
                    
                    # 计算最小差值方向
                    min_delta_lattice = delta_lattice - nearest_int
                    direction = self.geometry.to_euclidean_coords(min_delta_lattice)
                    
                    # 归一化
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        unit_direction = direction / norm
                        
                        # 梯度与误差成正比
                        grad = error * unit_direction
                        
                        # 节点u和v的梯度方向相反
                        gradients[u] += grad
                        gradients[v] -= grad
            
            # 应用梯度更新
            nodes -= learning_rate * gradients
            
            # 将节点包裹回基本域
            for i in range(n_nodes):
                wrapped, _ = self.geometry.wrap_to_fundamental_domain(nodes[i])
                nodes[i] = wrapped
        
        # 重新计算边（因为节点移动后，有些边可能不再满足距离条件）
        new_edges = []
        for u, v in edges:
            dist = self.geometry.get_metric(nodes[u], nodes[v])
            if abs(dist - 1.0) < epsilon:
                new_edges.append((u, v))
        
        # 创建新图
        relaxed_graph = UnitDistanceGraph(
            nodes=nodes,
            edges=new_edges,
            epsilon=epsilon,
            metadata={
                **graph.metadata,
                "relaxed": True,
                "relaxation_iterations": iterations,
                "learning_rate": learning_rate
            }
        )
        
        # 记录到历史
        self.history.append({
            "operation": "relaxation",
            "iteration": iteration if 'iteration' in locals() else 0,
            "original_edges": len(edges),
            "new_edges": len(new_edges),
            "node_displacement": np.mean(np.linalg.norm(nodes - graph.nodes, axis=1))
        })
        
        return relaxed_graph
    
    def anneal_epsilon(self, graph: UnitDistanceGraph, 
                      decay_factor: float = 0.9,
                      min_epsilon: float = 0.001) -> UnitDistanceGraph:
        """
        退火epsilon：逐渐减小距离容差
        
        Args:
            graph: 当前图
            decay_factor: 衰减因子
            min_epsilon: 最小epsilon
            
        Returns:
            更新后的图
        """
        new_epsilon = max(graph.epsilon * decay_factor, min_epsilon)
        
        if abs(new_epsilon - graph.epsilon) < 1e-10:
            # epsilon没有变化
            return graph.copy()
        
        # 使用新的epsilon重新构建边
        self.builder.config.epsilon = new_epsilon
        new_graph = self.builder.construct_graph(graph.nodes)
        
        # 恢复原始epsilon设置
        self.builder.config.epsilon = graph.epsilon
        
        # 记录到历史
        self.history.append({
            "operation": "annealing",
            "old_epsilon": graph.epsilon,
            "new_epsilon": new_epsilon,
            "old_edges": len(graph.edges),
            "new_edges": len(new_graph.edges)
        })
        
        return new_graph
    
    def optimize_coloring_conflict(self, graph: UnitDistanceGraph,
                                 coloring: Dict[int, int],
                                 num_points: int = 5) -> UnitDistanceGraph:
        """
        通过添加多个点来优化图的染色冲突
        
        Args:
            graph: 当前图
            coloring: 当前染色方案
            num_points: 要添加的点数
            
        Returns:
            更新后的图
        """
        current_graph = graph.copy()
        
        for i in range(num_points):
            # 寻找难以染色的点
            new_point = self.find_hard_point(current_graph, coloring)
            
            # 添加点到图中
            current_graph = self.builder.add_node_with_edges(current_graph, new_point)
            
            # 可选：松弛节点位置
            if i % 2 == 0:  # 每添加两个点松弛一次
                current_graph = self.relax_nodes(current_graph, iterations=3)
        
        # 记录到历史
        self.history.append({
            "operation": "conflict_optimization",
            "points_added": num_points,
            "final_nodes": len(current_graph.nodes),
            "final_edges": len(current_graph.edges)
        })
        
        return current_graph
    
    def get_history_summary(self) -> Dict[str, Any]:
        """获取优化历史摘要"""
        if not self.history:
            return {"empty": True}
        
        summary = {
            "total_operations": len(self.history),
            "methods_used": {},
            "points_generated": 0,
            "relaxations_performed": 0,
            "annealings_performed": 0
        }
        
        for entry in self.history:
            if "method" in entry:
                method = entry["method"]
                summary["methods_used"][method] = summary["methods_used"].get(method, 0) + 1
                summary["points_generated"] += 1
            elif "operation" in entry:
                op = entry["operation"]
                if op == "relaxation":
                    summary["relaxations_performed"] += 1
                elif op == "annealing":
                    summary["annealings_performed"] += 1
        
        return summary
    
    def clear_history(self):
        """清除优化历史"""
        self.history = []

# 预配置的优化器
def create_optimizer(geometry: GeometryEngine, builder: GraphBuilder,
                    method: str = "constraint_based") -> GraphOptimizer:
    """创建预配置的优化器"""
    config = OptimizerConfig(method=method)
    return GraphOptimizer(geometry, builder, config)