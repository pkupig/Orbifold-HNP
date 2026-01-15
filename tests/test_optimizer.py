"""
优化器测试 - 完整实现
"""
import unittest
import numpy as np
from pathlib import Path
import sys

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.geometry_engine import GeometryEngine
from src.core.graph_builder import GraphBuilder, GraphConfig, UnitDistanceGraph
from src.core.sat_solver import SATSolver, SATConfig
from src.core.optimizer import GraphOptimizer, OptimizerConfig

class TestGraphOptimizer(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.geometry = GeometryEngine()
        self.builder = GraphBuilder(self.geometry, GraphConfig(epsilon=0.05))
        self.config = OptimizerConfig(
            method="constraint_based",
            num_candidates=100,
            relaxation_iterations=3,
            learning_rate=0.1
        )
        self.optimizer = GraphOptimizer(self.geometry, self.builder, self.config)
        
        # 创建测试图
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])
        self.graph = self.builder.construct_graph(nodes, use_kdtree=False)
        
        # 创建测试染色方案
        self.coloring = {0: 0, 1: 1, 2: 0, 3: 1}
    
    def test_optimizer_creation(self):
        """测试优化器创建"""
        self.assertEqual(self.optimizer.config.method, "constraint_based")
        self.assertEqual(self.optimizer.config.num_candidates, 100)
        self.assertEqual(self.optimizer.config.relaxation_iterations, 3)
        self.assertEqual(self.optimizer.config.learning_rate, 0.1)
    
    def test_find_hard_point_methods(self):
        """测试不同方法寻找难以染色的点"""
        methods = ["constraint_based", "energy_based"]
        
        for method in methods:
            with self.subTest(method=method):
                config = OptimizerConfig(method=method, num_candidates=50)
                optimizer = GraphOptimizer(self.geometry, self.builder, config)
                
                point = optimizer.find_hard_point(self.graph, self.coloring, method=method)
                
                # 应该返回一个点
                self.assertIsNotNone(point)
                self.assertEqual(point.shape, (2,))
                
                # 点应该在基本域内（或附近）
                lattice_coords = self.geometry.to_lattice_coords(point)
                # 允许小的数值误差
                self.assertTrue(np.all(lattice_coords >= -0.1))
                self.assertTrue(np.all(lattice_coords <= 1.1))
    
    def test_find_hard_point_no_coloring(self):
        """测试没有染色方案时寻找点"""
        # 空染色方案
        empty_coloring = {}
        
        point = self.optimizer.find_hard_point(self.graph, empty_coloring)
        
        # 应该返回一个随机点
        self.assertIsNotNone(point)
        self.assertEqual(point.shape, (2,))
    
    def test_random_point_generation(self):
        """测试随机点生成"""
        point = self.optimizer._random_point_in_fundamental_domain()
        
        self.assertIsNotNone(point)
        self.assertEqual(point.shape, (2,))
        
        # 应该在基本域内
        lattice_coords = self.geometry.to_lattice_coords(point)
        self.assertTrue(np.all(lattice_coords >= 0))
        self.assertTrue(np.all(lattice_coords < 1))
    
    def test_relax_nodes(self):
        """测试节点松弛"""
        # 创建带有"不完美"边的图
        nodes = np.array([
            [0.0, 0.0],
            [1.1, 0.0]  # 距离不是1
        ])
        
        # 手动创建边，假设它们是单位距离
        graph = UnitDistanceGraph(
            nodes=nodes,
            edges=[(0, 1)],
            epsilon=0.2  # 使用较大的epsilon
        )
        
        # 松弛节点
        relaxed = self.optimizer.relax_nodes(graph, iterations=5)
        
        # 节点应该移动
        self.assertFalse(np.allclose(relaxed.nodes, graph.nodes))
        
        # 边应该仍然存在
        self.assertEqual(relaxed.num_edges, 1)
        
        # 距离应该更接近1
        dist_before = self.geometry.get_metric(graph.nodes[0], graph.nodes[1])
        dist_after = self.geometry.get_metric(relaxed.nodes[0], relaxed.nodes[1])
        
        # 松弛后距离应该更接近1
        self.assertLess(abs(dist_after - 1.0), abs(dist_before - 1.0) + 0.01)
    
    def test_anneal_epsilon(self):
        """测试epsilon退火"""
        original_epsilon = self.graph.epsilon
        
        # 退火
        annealed = self.optimizer.anneal_epsilon(
            self.graph,
            decay_factor=0.5,
            min_epsilon=0.001
        )
        
        # epsilon应该减小
        self.assertLess(annealed.epsilon, original_epsilon)
        self.assertEqual(annealed.epsilon, max(original_epsilon * 0.5, 0.001))
        
        # 边数可能改变
        # 通常更小的epsilon会减少边数
        self.assertLessEqual(annealed.num_edges, self.graph.num_edges)
    
    def test_optimize_coloring_conflict(self):
        """测试染色冲突优化"""
        # 添加一些点来增加冲突
        optimized = self.optimizer.optimize_coloring_conflict(
            self.graph,
            self.coloring,
            num_points=2
        )
        
        # 图应该变大
        self.assertGreater(optimized.num_nodes, self.graph.num_nodes)
        self.assertGreaterEqual(optimized.num_edges, self.graph.num_edges)
    
    def test_history_tracking(self):
        """测试历史跟踪"""
        # 执行一些操作
        self.optimizer.find_hard_point(self.graph, self.coloring)
        self.optimizer.relax_nodes(self.graph)
        self.optimizer.anneal_epsilon(self.graph)
        
        # 获取历史摘要
        summary = self.optimizer.get_history_summary()
        
        self.assertGreater(summary["total_operations"], 0)
        self.assertGreater(summary["points_generated"], 0)
        self.assertGreater(summary["relaxations_performed"], 0)
        self.assertGreater(summary["annealings_performed"], 0)
        
        # 清除历史
        self.optimizer.clear_history()
        summary = self.optimizer.get_history_summary()
        self.assertTrue(summary.get("empty", False) or summary["total_operations"] == 0)
    
    def test_candidate_generation(self):
        """测试候选点生成"""
        candidates = self.optimizer._generate_candidate_points(10)
        
        self.assertEqual(len(candidates), 10)
        
        for candidate in candidates:
            self.assertEqual(candidate.shape, (2,))
    
    def test_hybrid_method(self):
        """测试混合方法"""
        config = OptimizerConfig(method="hybrid", num_candidates=50)
        optimizer = GraphOptimizer(self.geometry, self.builder, config)
        
        point = optimizer.find_hard_point(self.graph, self.coloring)
        
        self.assertIsNotNone(point)
        self.assertEqual(point.shape, (2,))
    
    def test_evolutionary_method(self):
        """测试进化方法"""
        config = OptimizerConfig(method="evolutionary", num_candidates=50)
        optimizer = GraphOptimizer(self.geometry, self.builder, config)
        
        point = optimizer.find_hard_point(self.graph, self.coloring)
        
        self.assertIsNotNone(point)
        self.assertEqual(point.shape, (2,))

class TestOptimizerConfig(unittest.TestCase):
    
    def test_config_creation(self):
        """测试配置创建"""
        config = OptimizerConfig(
            method="test_method",
            num_candidates=500,
            relaxation_iterations=10,
            learning_rate=0.05,
            energy_sigma=0.01,
            attraction_factor=0.2
        )
        
        self.assertEqual(config.method, "test_method")
        self.assertEqual(config.num_candidates, 500)
        self.assertEqual(config.relaxation_iterations, 10)
        self.assertEqual(config.learning_rate, 0.05)
        self.assertEqual(config.energy_sigma, 0.01)
        self.assertEqual(config.attraction_factor, 0.2)
    
    def test_config_validation(self):
        """测试配置验证"""
        # 无效的参数
        with self.assertRaises(ValueError):
            OptimizerConfig(num_candidates=0)
        
        with self.assertRaises(ValueError):
            OptimizerConfig(learning_rate=0)
        
        with self.assertRaises(ValueError):
            OptimizerConfig(energy_sigma=0)

if __name__ == '__main__':
    unittest.main()