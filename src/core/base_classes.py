"""
基础类定义 - 为系统提供统一的基类接口
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic
import numpy as np
import json
import pickle
from datetime import datetime
from pathlib import Path

T = TypeVar('T')
ConfigT = TypeVar('ConfigT', bound='BaseConfig')

class Serializable(ABC):
    """可序列化接口"""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        pass
    
    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> 'Serializable':
        """从字典创建"""
        pass
    
    def save(self, filepath: str):
        """保存到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Serializable':
        """从文件加载"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls().from_dict(data)

@dataclass
class BaseConfig(Serializable):
    """基础配置类"""
    
    name: str = "unnamed_config"
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def from_dict(self, data: Dict[str, Any]) -> 'BaseConfig':
        """从字典创建"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def validate(self) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []
        # 基础验证
        if not self.name:
            errors.append("Config name cannot be empty")
        if not self.version:
            errors.append("Config version cannot be empty")
        return errors
    
    def update(self, updates: Dict[str, Any]):
        """更新配置"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)

@dataclass
class BaseResult(Serializable):
    """基础结果类"""
    
    success: bool = False
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 确保metadata是可JSON序列化的
        result['metadata'] = self._make_serializable(result['metadata'])
        return result
    
    def from_dict(self, data: Dict[str, Any]) -> 'BaseResult':
        """从字典创建"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def _make_serializable(self, obj: Any) -> Any:
        """使对象可JSON序列化"""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return str(obj)

class BaseModule(ABC, Generic[ConfigT]):
    """基础模块类"""
    
    def __init__(self, config: ConfigT):
        self.config = config
        self._initialized = False
        self._name = self.__class__.__name__
        
    def initialize(self) -> bool:
        """初始化模块"""
        try:
            if not self._initialized:
                self._initialize()
                self._initialized = True
                return True
            return False
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {self._name}: {str(e)}")
    
    @abstractmethod
    def _initialize(self):
        """子类实现的初始化逻辑"""
        pass
    
    def reset(self):
        """重置模块"""
        self._initialized = False
    
    def get_state(self) -> Dict[str, Any]:
        """获取模块状态"""
        return {
            "name": self._name,
            "initialized": self._initialized,
            "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
        }
    
    def load_state(self, state: Dict[str, Any]):
        """加载模块状态"""
        self._initialized = state.get("initialized", False)
    
    def __repr__(self) -> str:
        return f"{self._name}(initialized={self._initialized})"

class BaseAlgorithm(ABC):
    """基础算法类"""
    
    def __init__(self, name: str = "unnamed_algorithm"):
        self.name = name
        self._history: List[Dict[str, Any]] = []
        self._metrics: Dict[str, List[float]] = {}
        
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """运行算法"""
        pass
    
    def record_step(self, step_name: str, data: Dict[str, Any]):
        """记录算法步骤"""
        step_data = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        self._history.append(step_data)
    
    def record_metric(self, name: str, value: float):
        """记录指标"""
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(value)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取历史记录"""
        return self._history.copy()
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """获取指标"""
        return {k: v.copy() for k, v in self._metrics.items()}
    
    def clear_history(self):
        """清除历史记录"""
        self._history.clear()
        self._metrics.clear()

@dataclass
class Vector2D:
    """2D向量类"""
    
    x: float = 0.0
    y: float = 0.0
    
    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)
    
    @property
    def array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.x, self.y])
    
    @property
    def tuple(self) -> Tuple[float, float]:
        """转换为元组"""
        return (self.x, self.y)
    
    @property
    def magnitude(self) -> float:
        """向量模长"""
        return np.sqrt(self.x**2 + self.y**2)
    
    def normalized(self) -> 'Vector2D':
        """归一化向量"""
        mag = self.magnitude
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)
    
    def dot(self, other: 'Vector2D') -> float:
        """点积"""
        return self.x * other.x + self.y * other.y
    
    def cross(self, other: 'Vector2D') -> float:
        """叉积"""
        return self.x * other.y - self.y * other.x
    
    def distance_to(self, other: 'Vector2D') -> float:
        """到另一个向量的距离"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def rotate(self, angle_rad: float) -> 'Vector2D':
        """旋转向量"""
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        return Vector2D(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )
    
    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x / scalar, self.y / scalar)
    
    def __eq__(self, other: 'Vector2D') -> bool:
        return np.isclose(self.x, other.x) and np.isclose(self.y, other.y)
    
    def __repr__(self) -> str:
        return f"Vector2D({self.x:.6f}, {self.y:.6f})"
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Vector2D':
        return cls(data["x"], data["y"])

@dataclass
class BoundingBox:
    """边界框类"""
    
    min_x: float = 0.0
    min_y: float = 0.0
    max_x: float = 1.0
    max_y: float = 1.0
    
    @property
    def width(self) -> float:
        """宽度"""
        return self.max_x - self.min_x
    
    @property
    def height(self) -> float:
        """高度"""
        return self.max_y - self.min_y
    
    @property
    def center(self) -> Vector2D:
        """中心点"""
        return Vector2D(
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2
        )
    
    @property
    def area(self) -> float:
        """面积"""
        return self.width * self.height
    
    def contains(self, point: Union[Vector2D, Tuple[float, float]]) -> bool:
        """检查点是否在边界框内"""
        if isinstance(point, Vector2D):
            x, y = point.x, point.y
        else:
            x, y = point
        
        return (self.min_x <= x <= self.max_x and 
                self.min_y <= y <= self.max_y)
    
    def expand(self, amount: float) -> 'BoundingBox':
        """扩展边界框"""
        return BoundingBox(
            self.min_x - amount,
            self.min_y - amount,
            self.max_x + amount,
            self.max_y + amount
        )
    
    def intersect(self, other: 'BoundingBox') -> Optional['BoundingBox']:
        """计算交集"""
        min_x = max(self.min_x, other.min_x)
        min_y = max(self.min_y, other.min_y)
        max_x = min(self.max_x, other.max_x)
        max_y = min(self.max_y, other.max_y)
        
        if min_x < max_x and min_y < max_y:
            return BoundingBox(min_x, min_y, max_x, max_y)
        return None
    
    def union(self, other: 'BoundingBox') -> 'BoundingBox':
        """计算并集"""
        return BoundingBox(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y)
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "min_x": self.min_x,
            "min_y": self.min_y,
            "max_x": self.max_x,
            "max_y": self.max_y
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BoundingBox':
        return cls(**data)

# 类型别名
Point = Union[Vector2D, Tuple[float, float], np.ndarray]
Color = Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]
PathLike = Union[str, Path]