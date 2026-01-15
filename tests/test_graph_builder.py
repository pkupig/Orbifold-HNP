"""
图构建器测试 - 完整实现
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
from src.core.graph_builder import GraphBuilder, GraphConfig, UnitDistanceGraph

class TestGraphBuilder(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.geometry = GeometryEngine()
        self.config = GraphConfig(epsilon=0.05, use_kdtree=False)
        self.builder = GraphBuilder(self.geometry, self.config)
    
    def test_initialize_points(self):
        """测试点初始化"""
        n_points = 10
        
        # 测试不同采样方法
        methods = ["fibonacci", "grid", "random"]
        
        for method in methods:
            with self.subTest(method=method):
                points = self.builder.initialize_points(n_points, method=method)
                
                self.assertEqual(points.shape, (n_points, 2))
                
                # 检查点是否在基本域内
                for point in points:
                    lattice_coords = self.geometry.to_lattice_coords(point)
                    self.assertTrue(np.all(lattice_coords >= 0))
                    self.assertTrue(np.all(lattice_coords < 1))
    
    def test_construct_graph(self):
        """测试图构建"""
        # 创建已知的图（正方形）
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])
        
        # 构建图
        graph = self.builder.construct_graph(nodes, use_kdtree=False)
        
        # 检查基本属性
        self.assertEqual(graph.num_nodes, 4)
        self.assertGreater(graph.num_edges, 0)
        self.assertEqual(graph.epsilon, self.config.epsilon)
        
        # 检查边
        for u, v in graph.edges:
            self.assertLess(u, graph.num_nodes)
            self.assertLess(v, graph.num_nodes)
            self.assertNotEqual(u, v)
    
    def test_edge_construction(self):
        """测试边构建"""
        # 创建两个距离为1的点
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0]  # 距离为1
        ])
        
        # 使用较小的epsilon
        self.builder.config.epsilon = 0.01
        graph = self.builder.construct_graph(nodes, use_kdtree=False)
        
        # 应该有一条边
        self.assertEqual(graph.num_edges, 1)
        self.assertEqual(graph.edges[0], (0, 1))
    
    def test_edge_with_epsilon(self):
        """测试epsilon对边构建的影响"""
        nodes = np.array([
            [0.0, 0.0],
            [1.1, 0.0]  # 距离为1.1
        ])
        
        # 使用较大的epsilon
        self.builder.config.epsilon = 0.2
        graph1 = self.builder.construct_graph(nodes, use_kdtree=False)
        
        # 使用较小的epsilon
        self.builder.config.epsilon = 0.05
        graph2 = self.builder.construct_graph(nodes, use_kdtree=False)
        
        # 第一个图应该有边，第二个图应该没有
        self.assertGreater(graph1.num_edges, 0)
        self.assertEqual(graph2.num_edges, 0)
    
    def test_kdtree_vs_naive(self):
        """测试KDTree与朴素方法的比较"""
        n_points = 20
        nodes = self.builder.initialize_points(n_points, method="random")
        
        # 使用朴素方法
        graph_naive = self.builder.construct_graph(nodes, use_kdtree=False)
        
        # 使用KDTree
        self.builder.config.use_kdtree = True
        graph_kdtree = self.builder.construct_graph(nodes, use_kdtree=True)
        
        # 两个图应该有相同的节点和边
        self.assertEqual(graph_naive.num_nodes, graph_kdtree.num_nodes)
        
        # 边可能因为浮点误差略有不同，但应该大致相同
        edge_diff = abs(graph_naive.num_edges - graph_kdtree.num_edges)
        self.assertLess(edge_diff, max(graph_naive.num_edges, graph_kdtree.num_edges) * 0.1)
    
    def test_add_node(self):
        """测试添加节点"""
        # 创建初始图
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0]
        ])
        graph = self.builder.construct_graph(nodes, use_kdtree=False)
        
        # 添加新节点
        new_node = np.array([0.0, 1.0])
        new_graph = self.builder.add_node_with_edges(graph, new_node)
        
        # 检查新图
        self.assertEqual(new_graph.num_nodes, 3)
        self.assertGreaterEqual(new_graph.num_edges, graph.num_edges)
        
        # 新节点应该与某些现有节点连接
        new_node_edges = [edge for edge in new_graph.edges 
                         if 2 in edge]  # 新节点的索引是2
        self.assertGreater(len(new_node_edges), 0)
    
    def test_filter_sparse_nodes(self):
        """测试稀疏节点过滤"""
        # 创建包含孤立节点的图
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0],  # 与第一个节点连接
            [2.0, 2.0]   # 孤立节点
        ])
        
        graph = self.builder.construct_graph(nodes[:2], use_kdtree=False)
        
        # 添加孤立节点
        graph = UnitDistanceGraph(
            nodes=nodes,
            edges=[(0, 1)],  # 只有一条边
            epsilon=self.config.epsilon
        )
        
        # 过滤度数小于2的节点
        filtered = self.builder.filter_sparse_nodes(graph, min_degree=2)
        
        # 孤立节点应该被过滤掉
        self.assertEqual(filtered.num_nodes, 0)  # 没有节点度数为2
        
        # 过滤度数小于1的节点
        filtered = self.builder.filter_sparse_nodes(graph, min_degree=1)
        self.assertEqual(filtered.num_nodes, 2)  # 两个节点度数为1
    
    def test_merge_graphs(self):
        """测试图合并"""
        # 创建两个图
        nodes1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        nodes2 = np.array([[0.0, 1.0], [1.0, 1.0]])
        
        graph1 = self.builder.construct_graph(nodes1, use_kdtree=False)
        graph2 = self.builder.construct_graph(nodes2, use_kdtree=False)
        
        # 合并图
        merged = self.builder.merge_graphs(graph1, graph2)
        
        # 检查合并结果
        self.assertEqual(merged.num_nodes, graph1.num_nodes + graph2.num_nodes)
        self.assertGreaterEqual(merged.num_edges, graph1.num_edges + graph2.num_edges)
    
    def test_graph_properties(self):
        """测试图属性"""
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        graph = self.builder.construct_graph(nodes, use_kdtree=False)
        
        # 测试属性
        self.assertEqual(graph.num_nodes, 3)
        self.assertGreater(graph.num_edges, 0)
        self.assertGreaterEqual(graph.edge_density, 0)
        self.assertLessEqual(graph.edge_density, 1)
        
        # 测试度计算
        degrees = graph.get_node_degrees()
        self.assertEqual(len(degrees), graph.num_nodes)
        self.assertTrue(np.all(degrees >= 0))
    
    def test_to_networkx(self):
        """测试转换为NetworkX图"""
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0]
        ])
        
        graph = self.builder.construct_graph(nodes, use_kdtree=False)
        nx_graph = graph.to_networkx()
        
        # 检查NetworkX图属性
        self.assertEqual(nx_graph.number_of_nodes(), graph.num_nodes)
        self.assertEqual(nx_graph.number_of_edges(), graph.num_edges)
        
        # 检查位置属性
        for i in range(graph.num_nodes):
            self.assertIn('pos', nx_graph.nodes[i])
            np.testing.assert_array_almost_equal(
                nx_graph.nodes[i]['pos'],
                graph.nodes[i]
            )
    
    def test_save_load_graph(self):
        """测试图保存和加载"""
        # 创建测试图
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        graph = self.builder.construct_graph(nodes, use_kdtree=False)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            graph_file = f.name
            
            try:
                # 保存图
                self.builder.save_graph(graph, graph_file)
                
                # 加载图
                loaded_graph = GraphBuilder.load_graph(graph_file)
                
                # 检查图相同
                self.assertEqual(loaded_graph.num_nodes, graph.num_nodes)
                self.assertEqual(loaded_graph.num_edges, graph.num_edges)
                self.assertEqual(loaded_graph.epsilon, graph.epsilon)
                
                np.testing.assert_array_almost_equal(
                    loaded_graph.nodes,
                    graph.nodes
                )
                
                # 边可能顺序不同，但内容相同
                self.assertEqual(
                    set(loaded_graph.edges),
                    set(graph.edges)
                )
                
            finally:
                os.unlink(graph_file)

class TestUnitDistanceGraph(unittest.TestCase):
    
    def test_graph_creation(self):
        """测试图创建"""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        edges = [(0, 1)]
        epsilon = 0.05
        
        graph = UnitDistanceGraph(nodes=nodes, edges=edges, epsilon=epsilon)
        
        self.assertEqual(graph.num_nodes, 2)
        self.assertEqual(graph.num_edges, 1)
        self.assertEqual(graph.epsilon, epsilon)
    
    def test_invalid_graph(self):
        """测试无效图创建"""
        # 无效的节点形状
        with self.assertRaises(ValueError):
            UnitDistanceGraph(
                nodes=np.array([0.0, 0.0]),  # 应该是2D数组
                edges=[],
                epsilon=0.05
            )
        
        # 自环
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        edges = [(0, 0)]  # 自环
        with self.assertRaises(ValueError):
            UnitDistanceGraph(nodes=nodes, edges=edges, epsilon=0.05)
    
    def test_edge_ordering(self):
        """测试边排序"""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        edges = [(2, 0), (1, 0)]  # 乱序
        
        graph = UnitDistanceGraph(nodes=nodes, edges=edges, epsilon=0.05)
        
        # 边应该被排序
        for u, v in graph.edges:
            self.assertLess(u, v)
    
    def test_add_nodes(self):
        """测试添加节点"""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        edges = [(0, 1)]
        graph = UnitDistanceGraph(nodes=nodes, edges=edges, epsilon=0.05)
        
        # 添加单个节点
        new_graph = graph.add_node(np.array([0.0, 1.0]))
        self.assertEqual(new_graph.num_nodes, 3)
        self.assertEqual(new_graph.num_edges, 1)  # 边不变
        
        # 添加多个节点
        new_nodes = np.array([[0.5, 0.5], [0.7, 0.3]])
        new_graph = graph.add_nodes(new_nodes)
        self.assertEqual(new_graph.num_nodes, 4)
    
    def test_copy(self):
        """测试图复制"""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        edges = [(0, 1)]
        graph = UnitDistanceGraph(nodes=nodes, edges=edges, epsilon=0.05)
        
        # 复制图
        graph_copy = graph.copy()
        
        # 修改原始图
        graph.nodes[0] = [2.0, 2.0]
        graph.edges.append((0, 1))  # 重复边，会被去重
        
        # 复制图不应该改变
        self.assertNotEqual(graph_copy.nodes[0, 0], 2.0)
        self.assertEqual(graph_copy.num_edges, 1)

if __name__ == '__main__':
    unittest.main()