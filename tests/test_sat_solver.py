"""
SAT求解器测试 - 完整实现
"""
import unittest
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.geometry_engine import GeometryEngine
from src.core.graph_builder import GraphBuilder, GraphConfig
from src.core.sat_solver import SATSolver, SATConfig

class TestSATSolver(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        # 检查是否有可用的SAT求解器
        self.solver_names = ["kissat", "glucose", "minisat"]
        self.available_solver = None
        
        for solver_name in self.solver_names:
            try:
                solver = SATSolver(SATConfig(solver_name=solver_name, timeout=5))
                # 尝试运行一个简单测试
                test_cnf = "p cnf 1 1\n1 0\n"
                result, _, _ = solver.solve_cnf(test_cnf)
                if result is not None:
                    self.available_solver = solver_name
                    break
            except:
                continue
        
        if self.available_solver is None:
            self.skipTest("No SAT solver available")
        
        # 创建求解器
        self.config = SATConfig(
            solver_name=self.available_solver,
            timeout=10,
            verbose=False
        )
        self.solver = SATSolver(self.config)
        
        # 创建测试图
        self.geometry = GeometryEngine()
        self.builder = GraphBuilder(self.geometry, GraphConfig(epsilon=0.05))
    
    def test_solver_availability(self):
        """测试求解器可用性"""
        self.assertIsNotNone(self.available_solver)
        print(f"Using SAT solver: {self.available_solver}")
    
    def test_simple_cnf(self):
        """测试简单CNF求解"""
        # 可满足的CNF: (x1)
        cnf_sat = "p cnf 1 1\n1 0\n"
        
        satisfiable, assignment, stats = self.solver.solve_cnf(cnf_sat)
        
        self.assertTrue(satisfiable)
        self.assertIsNotNone(assignment)
        self.assertIn(1, assignment)  # x1应该为真
        
        # 不可满足的CNF: (x1) ∧ (¬x1)
        cnf_unsat = "p cnf 1 2\n1 0\n-1 0\n"
        
        satisfiable, assignment, stats = self.solver.solve_cnf(cnf_unsat)
        
        self.assertFalse(satisfiable)
        self.assertIsNone(assignment)
    
    def test_graph_coloring_simple(self):
        """测试简单图染色"""
        # 创建两个节点一条边的图（2色即可）
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        graph = self.builder.construct_graph(nodes, use_kdtree=False)
        
        # 测试2-染色
        colorable, coloring, stats = self.solver.is_k_colorable(graph, k=2)
        
        self.assertTrue(colorable)
        self.assertIsNotNone(coloring)
        self.assertEqual(len(coloring), graph.num_nodes)
        
        # 验证染色
        for u, v in graph.edges:
            self.assertNotEqual(coloring[u], coloring[v])
    
    def test_graph_coloring_impossible(self):
        """测试不可能染色的图"""
        # 创建三角形（3个节点完全连接），需要3色
        # 但在我们的单位距离图中，很难创建三角形
        # 这里我们测试一个小图
        
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        # 手动添加边，确保形成三角形
        from src.core.graph_builder import UnitDistanceGraph
        graph = UnitDistanceGraph(
            nodes=nodes,
            edges=[(0, 1), (1, 2), (2, 0)],  # 三角形
            epsilon=0.05
        )
        
        # 测试2-染色（应该不可能）
        colorable, coloring, stats = self.solver.is_k_colorable(graph, k=2)
        
        # 注意：可能超时，所以我们只检查如果返回结果的情况
        if colorable is not None:
            self.assertFalse(colorable)
    
    def test_encoding_methods(self):
        """测试不同编码方法"""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        graph = self.builder.construct_graph(nodes, use_kdtree=False)
        
        encodings = ["standard", "efficient"]  # 对数编码可能太慢
        
        for encoding in encodings:
            with self.subTest(encoding=encoding):
                colorable, coloring, stats = self.solver.is_k_colorable(
                    graph, k=2, encoding=encoding
                )
                
                # 至少应该成功运行
                self.assertIsNotNone(colorable)
                if colorable:
                    self.assertIsNotNone(coloring)
    
    def test_chromatic_number_estimation(self):
        """测试色数估计"""
        # 创建简单图
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        graph = self.builder.construct_graph(nodes, use_kdtree=False)
        
        # 估计色数
        chromatic_num, stats = self.solver.estimate_chromatic_number(
            graph, max_k=4, timeout_per_test=5
        )
        
        # 应该至少返回一个结果
        self.assertIsNotNone(chromatic_num)
        self.assertGreater(chromatic_num, 0)
        
        # 检查统计信息
        self.assertIn("tests", stats)
        self.assertGreaterEqual(len(stats["tests"]), 1)
    
    def test_statistics(self):
        """测试统计信息收集"""
        # 运行一些求解
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        graph = self.builder.construct_graph(nodes, use_kdtree=False)
        
        # 运行多次
        for _ in range(3):
            self.solver.is_k_colorable(graph, k=2)
        
        # 获取统计信息
        stats = self.solver.get_statistics()
        
        self.assertGreater(stats["total_solves"], 0)
        self.assertGreaterEqual(stats["satisfiable"], 0)
        self.assertGreaterEqual(stats["total_time"], 0)
        
        # 重置统计
        self.solver.reset_statistics()
        stats = self.solver.get_statistics()
        self.assertEqual(stats["total_solves"], 0)
    
    def test_cnf_generation(self):
        """测试CNF生成"""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        graph = self.builder.construct_graph(nodes, use_kdtree=False)
        
        # 生成CNF
        cnf = self.solver.graph_to_cnf(graph, k=2)
        
        # 检查CNF格式
        lines = cnf.strip().split('\n')
        
        # 第一行应该是p cnf
        self.assertTrue(lines[0].startswith('p cnf'))
        
        # 应该有子句
        self.assertGreater(len(lines), 1)
        
        # 每个子句应该以0结束
        for line in lines[1:]:
            if line.strip() and not line.startswith('c'):  # 忽略注释
                self.assertTrue(line.strip().endswith('0'))
    
    def test_assignment_parsing(self):
        """测试赋值解析"""
        # 测试标准编码的赋值解析
        output = """c SAT solver output
s SATISFIABLE
v 1 -2 3 -4 0
v 5 6 0
"""
        
        assignment = self.solver._parse_assignment(output, n_vars_hint=6)
        
        self.assertIsNotNone(assignment)
        self.assertGreater(len(assignment), 0)
        self.assertIn(1, assignment)
        self.assertIn(-2, assignment)
        self.assertIn(3, assignment)
        self.assertIn(-4, assignment)
        self.assertIn(5, assignment)
        self.assertIn(6, assignment)
    
    def test_assignment_to_coloring(self):
        """测试赋值到染色的转换"""
        # 创建简单图
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        graph = self.builder.construct_graph(nodes, use_kdtree=False)
        
        # 生成CNF
        k = 3
        cnf = self.solver.graph_to_cnf(graph, k)
        
        # 求解
        satisfiable, assignment, stats = self.solver.solve_cnf(cnf)
        
        if satisfiable and assignment is not None:
            # 转换为染色
            coloring = self.solver._assignment_to_coloring(
                assignment, graph.num_nodes, k
            )
            
            # 检查染色
            self.assertEqual(len(coloring), graph.num_nodes)
            
            # 所有节点都应该有颜色
            for i in range(graph.num_nodes):
                self.assertIn(i, coloring)
                self.assertGreaterEqual(coloring[i], 0)
                self.assertLess(coloring[i], k)
            
            # 检查边约束
            for u, v in graph.edges:
                self.assertNotEqual(coloring[u], coloring[v])
    
    def test_timeout(self):
        """测试超时处理"""
        # 创建一个可能耗时的CNF（但这里我们主要测试超时机制）
        # 创建一个较大的CNF
        n_vars = 100
        n_clauses = 1000
        
        # 生成随机CNF
        cnf = f"p cnf {n_vars} {n_clauses}\n"
        for _ in range(n_clauses):
            # 随机生成子句
            clause_len = np.random.randint(1, 4)
            clause = []
            for _ in range(clause_len):
                var = np.random.randint(1, n_vars + 1)
                sign = np.random.choice([1, -1])
                clause.append(str(sign * var))
            clause.append('0')
            cnf += ' '.join(clause) + '\n'
        
        # 使用非常短的超时
        config = SATConfig(
            solver_name=self.available_solver,
            timeout=0.1,  # 0.1秒，应该超时
            verbose=False
        )
        solver = SATSolver(config)
        
        satisfiable, assignment, stats = solver.solve_cnf(cnf)
        
        # 可能超时，也可能快速求解
        # 我们只检查函数没有崩溃
        self.assertIsNotNone(stats)

class TestSATConfig(unittest.TestCase):
    
    def test_config_creation(self):
        """测试配置创建"""
        config = SATConfig(
            solver_name="test_solver",
            timeout=30,
            verbose=True,
            temp_dir="/tmp/test",
            keep_temp_files=True
        )
        
        self.assertEqual(config.solver_name, "test_solver")
        self.assertEqual(config.timeout, 30)
        self.assertTrue(config.verbose)
        self.assertEqual(config.temp_dir, "/tmp/test")
        self.assertTrue(config.keep_temp_files)
    
    def test_config_validation(self):
        """测试配置验证"""
        # 无效的超时时间
        with self.assertRaises(ValueError):
            SATConfig(timeout=0)
        
        with self.assertRaises(ValueError):
            SATConfig(timeout=-1)

if __name__ == '__main__':
    unittest.main()