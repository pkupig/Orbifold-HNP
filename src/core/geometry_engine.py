"""
几何引擎模块 - 完整的实现
处理轨形上的距离计算和坐标变换
"""
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass, field
import warnings
import json

@dataclass
class LatticeConfig:
    """晶格配置类"""
    type: str = "hexagonal"
    v1: Tuple[float, float] = (1.0, 0.0)
    v2: Tuple[float, float] = (0.5, np.sqrt(3)/2)
    wrap_to_fundamental: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        self.v1 = tuple(float(x) for x in self.v1)
        self.v2 = tuple(float(x) for x in self.v2)
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type,
            "v1": list(self.v1),
            "v2": list(self.v2),
            "wrap_to_fundamental": self.wrap_to_fundamental
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LatticeConfig':
        """从字典创建"""
        return cls(**data)

class GeometryEngine:
    """
    几何引擎：处理轨形上的距离计算和坐标变换
    支持多种晶格类型：六边形、正方形等
    """
    
    def __init__(self, config: Optional[LatticeConfig] = None):
        """
        初始化几何引擎
        
        Args:
            config: 晶格配置，如果为None则使用默认六边形晶格
        """
        self.config = config or LatticeConfig()
        self._setup_lattice()
        
    def _setup_lattice(self):
        """设置晶格矩阵和变换"""
        # 设置基础向量
        self.v1 = np.array(self.config.v1, dtype=np.float64)
        self.v2 = np.array(self.config.v2, dtype=np.float64)
        
        # 计算晶格矩阵和逆矩阵
        self.lattice_matrix = np.column_stack([self.v1, self.v2])
        self.inv_lattice_matrix = np.linalg.inv(self.lattice_matrix)
        
        # 计算晶格参数
        self.det = np.linalg.det(self.lattice_matrix)
        self.area = abs(self.det)  # 基本域面积
        
        # 检查晶格是否有效
        if abs(self.det) < 1e-10:
            raise ValueError("Lattice vectors are linearly dependent")
        
        # 计算晶格对偶基向量（用于距离计算）
        self.dual_v1 = 2 * np.pi * np.array([self.v2[1], -self.v2[0]]) / self.det
        self.dual_v2 = 2 * np.pi * np.array([-self.v1[1], self.v1[0]]) / self.det
        
    def to_lattice_coords(self, point: np.ndarray) -> np.ndarray:
        """
        将欧几里得坐标转换为晶格坐标
        
        Args:
            point: 欧几里得坐标 [x, y]
            
        Returns:
            晶格坐标 [u, v]
        """
        return self.inv_lattice_matrix @ point
    
    def to_euclidean_coords(self, lattice_coords: np.ndarray) -> np.ndarray:
        """
        将晶格坐标转换为欧几里得坐标
        
        Args:
            lattice_coords: 晶格坐标 [u, v]
            
        Returns:
            欧几里得坐标 [x, y]
        """
        return self.lattice_matrix @ lattice_coords
    
    def wrap_to_fundamental_domain(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将点包裹到基本域 [0,1) × [0,1) 中（晶格坐标下）
        
        Args:
            point: 欧几里得坐标
            
        Returns:
            (wrapped_point, translation_vector)
        """
        if not self.config.wrap_to_fundamental:
            return point.copy(), np.zeros(2)
        
        # 转换为晶格坐标
        lattice_coords = self.to_lattice_coords(point)
        
        # 取小数部分，将点映射到基本域
        fractional = lattice_coords - np.floor(lattice_coords)
        
        # 转换回欧几里得坐标
        wrapped_point = self.to_euclidean_coords(fractional)
        
        # 计算平移向量
        integer_part = np.floor(lattice_coords)
        translation = self.to_euclidean_coords(integer_part)
        
        return wrapped_point, translation
    
    def get_metric(self, point_u: np.ndarray, point_v: np.ndarray) -> float:
        """
        计算商空间 T² = ℝ² / Λ 上的距离
        考虑所有晶格平移后的最小距离
        
        Args:
            point_u: 第一个点
            point_v: 第二个点
            
        Returns:
            商空间中的最小距离
        """
        # 计算差值
        delta = point_v - point_u
        
        # 转换为晶格坐标的差值
        delta_lattice = self.to_lattice_coords(delta)
        
        # 找到最近的整数平移
        nearest_int = np.round(delta_lattice)
        
        # 计算在晶格坐标下的最小差值
        min_delta_lattice = delta_lattice - nearest_int
        
        # 转换回欧几里得距离
        min_delta_euclidean = self.to_euclidean_coords(min_delta_lattice)
        
        return np.linalg.norm(min_delta_euclidean)
    
    def get_periodic_distance(self, p1: np.ndarray, p2: np.ndarray, 
                             max_shifts: int = 1) -> Tuple[float, np.ndarray]:
        """
        计算周期距离并返回最小平移
        
        Args:
            p1: 第一个点
            p2: 第二个点
            max_shifts: 最大平移搜索范围
            
        Returns:
            (最小距离, 平移向量)
        """
        min_dist = float('inf')
        best_shift = np.zeros(2)
        
        # 在晶格平移范围内搜索
        for i in range(-max_shifts, max_shifts + 1):
            for j in range(-max_shifts, max_shifts + 1):
                shift = self.to_euclidean_coords(np.array([i, j]))
                shifted_p2 = p2 + shift
                dist = np.linalg.norm(p1 - shifted_p2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_shift = shift
        
        return min_dist, best_shift
    
    def get_nearby_lattice_points(self, point: np.ndarray, 
                                 search_radius: int = 1) -> List[np.ndarray]:
        """
        获取点附近的所有晶格点
        
        Args:
            point: 参考点
            search_radius: 搜索半径（晶格单位）
            
        Returns:
            附近晶格点的列表
        """
        nearby_points = []
        
        # 将点转换为晶格坐标
        lattice_coords = self.to_lattice_coords(point)
        base_int = np.floor(lattice_coords).astype(int)
        
        # 搜索周围的晶格点
        for i in range(-search_radius, search_radius + 1):
            for j in range(-search_radius, search_radius + 1):
                shift = np.array([i, j])
                lattice_point = base_int + shift
                euclidean_point = self.to_euclidean_coords(lattice_point)
                nearby_points.append(euclidean_point)
        
        return nearby_points
    
    def lift_to_cover(self, point: np.ndarray, 
                      num_copies: Tuple[int, int] = (3, 3)) -> List[np.ndarray]:
        """
        将基本域的点映射回 ℝ² 的局部覆盖
        
        Args:
            point: 基本域中的点
            num_copies: 在u和v方向上显示的副本数量
            
        Returns:
            覆盖空间中的点副本列表
        """
        copies = []
        
        # 将点包裹到基本域并获取基本平移
        wrapped_point, base_translation = self.wrap_to_fundamental_domain(point)
        
        # 在指定范围内生成副本
        nu, nv = num_copies
        for i in range(-nu//2, nu//2 + 1):
            for j in range(-nv//2, nv//2 + 1):
                shift = self.to_euclidean_coords(np.array([i, j]))
                copy_point = wrapped_point + base_translation + shift
                copies.append(copy_point)
        
        return copies
    
    def get_fundamental_domain_vertices(self) -> np.ndarray:
        """
        获取基本域的顶点
        
        Returns:
            顶点坐标数组 (n, 2)
        """
        if self.config.type == "hexagonal":
            # 六边形基本域（平行四边形）
            vertices = [
                np.array([0.0, 0.0]),
                self.v1,
                self.v1 + self.v2,
                self.v2
            ]
        else:  # square
            # 正方形基本域
            vertices = [
                np.array([0.0, 0.0]),
                self.v1,
                self.v1 + self.v2,
                self.v2
            ]
        
        return np.array(vertices)
    
    def generate_grid_points(self, resolution: Tuple[int, int]) -> np.ndarray:
        """
        在基本域内生成均匀网格点
        
        Args:
            resolution: 网格分辨率 (nu, nv)
            
        Returns:
            网格点坐标数组
        """
        nu, nv = resolution
        points = []
        
        for i in range(nu):
            for j in range(nv):
                # 均匀网格
                u = (i + 0.5) / nu
                v = (j + 0.5) / nv
                
                # 转换为欧几里得坐标
                point = self.to_euclidean_coords(np.array([u, v]))
                points.append(point)
        
        return np.array(points)
    
    def generate_fibonacci_points(self, n_points: int) -> np.ndarray:
        """
        使用Fibonacci晶格在基本域内生成确定性采样点
        
        Args:
            n_points: 点数
            
        Returns:
            点坐标数组
        """
        phi = (1 + np.sqrt(5)) / 2  # 黄金比例
        points = []
        
        for i in range(1, n_points + 1):
            # Fibonacci晶格
            u = (i / phi) % 1
            v = i / n_points
            
            # 转换为欧几里得坐标
            point = self.to_euclidean_coords(np.array([u, v]))
            points.append(point)
        
        return np.array(points)
    
    def compute_reciprocal_lattice(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算倒易晶格向量
        
        Returns:
            (倒易基向量1, 倒易基向量2)
        """
        # 倒易晶格向量
        b1 = 2 * np.pi * np.array([self.v2[1], -self.v2[0]]) / self.det
        b2 = 2 * np.pi * np.array([-self.v1[1], self.v1[0]]) / self.det
        
        return b1, b2
    
    def get_brillouin_zone(self) -> np.ndarray:
        """
        获取第一布里渊区
        
        Returns:
            布里渊区顶点坐标
        """
        # 计算倒易晶格向量
        b1, b2 = self.compute_reciprocal_lattice()
        
        # 布里渊区是倒易晶格的Voronoi图
        # 对于六边形晶格，布里渊区是六边形
        if self.config.type == "hexagonal":
            # 六边形布里渊区
            vertices = []
            for i in range(6):
                angle = 2 * np.pi * i / 6
                # 第一个倒易向量的长度
                b_norm = np.linalg.norm(b1)
                vertex = b_norm * np.array([np.cos(angle), np.sin(angle)])
                vertices.append(vertex)
            
            return np.array(vertices)
        else:
            # 正方形布里渊区
            return np.array([
                [-np.pi, -np.pi],
                [np.pi, -np.pi],
                [np.pi, np.pi],
                [-np.pi, np.pi]
            ])
    
    def save_config(self, filepath: str):
        """保存配置到文件"""
        config_dict = self.config.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'GeometryEngine':
        """从文件加载配置并创建几何引擎"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = LatticeConfig.from_dict(config_dict)
        return cls(config)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"GeometryEngine(lattice_type={self.config.type}, "
                f"v1={self.v1.tolist()}, v2={self.v2.tolist()})")

# 预定义的晶格配置
HEXAGONAL_LATTICE = LatticeConfig(
    type="hexagonal",
    v1=(1.0, 0.0),
    v2=(0.5, np.sqrt(3)/2)
)

SQUARE_LATTICE = LatticeConfig(
    type="square",
    v1=(1.0, 0.0),
    v2=(0.0, 1.0)
)

TRIANGULAR_LATTICE = LatticeConfig(
    type="triangular",
    v1=(1.0, 0.0),
    v2=(0.5, np.sqrt(3)/2)
)