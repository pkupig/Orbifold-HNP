"""
几何引擎测试 - 完整实现
"""
import unittest
import numpy as np
from pathlib import Path
import sys

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.geometry_engine import GeometryEngine, LatticeConfig

class TestGeometryEngine(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.hex_engine = GeometryEngine(
            LatticeConfig(type="hexagonal")
        )
        self.square_engine = GeometryEngine(
            LatticeConfig(type="square")
        )
    
    def test_lattice_vectors(self):
        """测试晶格向量"""
        # 六边形晶格
        self.assertEqual(self.hex_engine.config.type, "hexagonal")
        np.testing.assert_array_almost_equal(
            self.hex_engine.v1, 
            np.array([1.0, 0.0])
        )
        np.testing.assert_array_almost_equal(
            self.hex_engine.v2,
            np.array([0.5, np.sqrt(3)/2])
        )
        
        # 正方形晶格
        self.assertEqual(self.square_engine.config.type, "square")
        np.testing.assert_array_almost_equal(
            self.square_engine.v1,
            np.array([1.0, 0.0])
        )
        np.testing.assert_array_almost_equal(
            self.square_engine.v2,
            np.array([0.0, 1.0])
        )
    
    def test_coordinate_conversion(self):
        """测试坐标转换"""
        # 测试六边形晶格
        test_point = np.array([0.5, 0.5])
        
        # 转换为晶格坐标
        lattice_coords = self.hex_engine.to_lattice_coords(test_point)
        
        # 转换回欧几里得坐标
        euclidean_coords = self.hex_engine.to_euclidean_coords(lattice_coords)
        
        np.testing.assert_array_almost_equal(test_point, euclidean_coords)
    
    def test_wrap_to_fundamental_domain(self):
        """测试基本域包裹"""
        # 点在基本域外
        point = np.array([2.5, 1.8])
        
        wrapped, translation = self.hex_engine.wrap_to_fundamental_domain(point)
        
        # 检查包裹后的点在基本域内
        lattice_coords = self.hex_engine.to_lattice_coords(wrapped)
        self.assertTrue(np.all(lattice_coords >= 0))
        self.assertTrue(np.all(lattice_coords < 1))
        
        # 检查原始点 = 包裹点 + 平移
        reconstructed = wrapped + translation
        np.testing.assert_array_almost_equal(point, reconstructed)
    
    def test_get_metric(self):
        """测试距离计算"""
        # 测试六边形晶格中的距离
        point1 = np.array([0.0, 0.0])
        point2 = np.array([1.0, 0.0])
        
        distance = self.hex_engine.get_metric(point1, point2)
        self.assertAlmostEqual(distance, 1.0, places=10)
        
        # 测试周期性
        point3 = point2 + self.hex_engine.v1 + self.hex_engine.v2
        distance2 = self.hex_engine.get_metric(point1, point3)
        self.assertAlmostEqual(distance, distance2, places=10)
    
    def test_distance_symmetry(self):
        """测试距离对称性"""
        points = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ])
        
        for engine in [self.hex_engine, self.square_engine]:
            # 计算距离矩阵
            n = len(points)
            dist_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    dist_matrix[i, j] = engine.get_metric(points[i], points[j])
            
            # 检查对称性
            np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)
            
            # 检查对角线为0
            np.testing.assert_array_almost_equal(np.diag(dist_matrix), np.zeros(n))
    
    def test_periodic_distance(self):
        """测试周期距离"""
        point1 = np.array([0.1, 0.1])
        point2 = np.array([0.9, 0.9])  # 靠近基本域边界
        
        # 在正方形晶格中，周期距离应该很小
        dist, shift = self.square_engine.get_periodic_distance(point1, point2)
        
        self.assertLess(dist, 0.3)  # 应该小于0.3
        self.assertTrue(np.any(shift != 0))  # 应该有平移
    
    def test_fundamental_domain_vertices(self):
        """测试基本域顶点"""
        # 六边形晶格
        hex_vertices = self.hex_engine.get_fundamental_domain_vertices()
        self.assertEqual(hex_vertices.shape, (4, 2))
        
        # 正方形晶格
        square_vertices = self.square_engine.get_fundamental_domain_vertices()
        self.assertEqual(square_vertices.shape, (4, 2))
        
        # 检查顶点顺序（应该是逆时针）
        hex_area = self._polygon_area(hex_vertices)
        self.assertGreater(hex_area, 0)
        
        square_area = self._polygon_area(square_vertices)
        self.assertAlmostEqual(square_area, 1.0)
    
    def test_generate_points(self):
        """测试点生成"""
        # 测试Fibonacci点生成
        n_points = 10
        fib_points = self.hex_engine.generate_fibonacci_points(n_points)
        
        self.assertEqual(fib_points.shape, (n_points, 2))
        
        # 测试网格点生成
        grid_points = self.hex_engine.generate_grid_points((3, 3))
        self.assertEqual(grid_points.shape, (9, 2))
    
    def test_reciprocal_lattice(self):
        """测试倒易晶格"""
        b1, b2 = self.hex_engine.compute_reciprocal_lattice()
        
        # 检查正交关系: a_i · b_j = 2π δ_ij
        dot11 = np.dot(self.hex_engine.v1, b1)
        dot12 = np.dot(self.hex_engine.v1, b2)
        dot21 = np.dot(self.hex_engine.v2, b1)
        dot22 = np.dot(self.hex_engine.v2, b2)
        
        self.assertAlmostEqual(dot11, 2 * np.pi, places=10)
        self.assertAlmostEqual(dot12, 0, places=10)
        self.assertAlmostEqual(dot21, 0, places=10)
        self.assertAlmostEqual(dot22, 2 * np.pi, places=10)
    
    def _polygon_area(self, vertices):
        """计算多边形面积（用于测试）"""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i, 0] * vertices[j, 1]
            area -= vertices[j, 0] * vertices[i, 1]
        return abs(area) / 2.0
    
    def test_save_load_config(self):
        """测试配置保存和加载"""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_file = f.name
            
            try:
                # 保存配置
                self.hex_engine.save_config(config_file)
                
                # 加载配置
                loaded_engine = GeometryEngine.load_config(config_file)
                
                # 检查配置相同
                self.assertEqual(loaded_engine.config.type, self.hex_engine.config.type)
                np.testing.assert_array_almost_equal(
                    loaded_engine.v1, 
                    self.hex_engine.v1
                )
                np.testing.assert_array_almost_equal(
                    loaded_engine.v2,
                    self.hex_engine.v2
                )
                
            finally:
                os.unlink(config_file)

if __name__ == '__main__':
    unittest.main()