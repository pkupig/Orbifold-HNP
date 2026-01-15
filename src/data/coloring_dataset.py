"""
染色数据集 - 完整实现
管理和操作染色方案的数据集
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import hashlib
from collections import defaultdict

from ..core.graph_builder import UnitDistanceGraph
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class Coloring:
    """染色方案"""
    colors: Dict[int, int]  # 节点索引 -> 颜色索引
    k: int  # 使用的颜色数
    is_valid: bool = True
    conflicts: List[Tuple[int, int]] = field(default_factory=list)  # 冲突边列表
    score: float = 0.0  # 评分（越高越好）
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "colors": self.colors,
            "k": self.k,
            "is_valid": self.is_valid,
            "conflicts": self.conflicts,
            "score": self.score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Coloring':
        return cls(**data)
    
    @property
    def num_colors_used(self) -> int:
        """实际使用的颜色数"""
        return len(set(self.colors.values()))
    
    def get_color_distribution(self) -> Dict[int, int]:
        """获取颜色分布"""
        distribution = defaultdict(int)
        for color in self.colors.values():
            distribution[color] += 1
        return dict(distribution)

@dataclass
class ColoringMetadata:
    """染色元数据"""
    graph_id: str
    method: str = "unknown"
    solver: str = "unknown"
    runtime: float = 0.0
    optimal: bool = False
    bounds: Tuple[int, int] = (1, 100)  # 色数下界和上界
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "method": self.method,
            "solver": self.solver,
            "runtime": self.runtime,
            "optimal": self.optimal,
            "bounds": list(self.bounds),
            "properties": self.properties,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColoringMetadata':
        data = data.copy()
        data["bounds"] = tuple(data["bounds"])
        return cls(**data)

@dataclass
class ColoringRecord:
    """染色记录"""
    coloring_id: str
    coloring: Coloring
    metadata: ColoringMetadata
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def update(self):
        """更新记录时间"""
        from datetime import datetime
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "coloring_id": self.coloring_id,
            "coloring": self.coloring.to_dict(),
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColoringRecord':
        coloring = Coloring.from_dict(data["coloring"])
        metadata = ColoringMetadata.from_dict(data["metadata"])
        
        return cls(
            coloring_id=data["coloring_id"],
            coloring=coloring,
            metadata=metadata,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", "")
        )

@dataclass
class ColoringDatasetConfig:
    """染色数据集配置"""
    name: str = "coloring_dataset"
    description: str = ""
    max_size: int = 5000
    auto_save: bool = True
    save_interval: int = 20
    compression: bool = True
    index_properties: List[str] = field(default_factory=lambda: ["graph_id", "method", "k"])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "max_size": self.max_size,
            "auto_save": self.auto_save,
            "save_interval": self.save_interval,
            "compression": self.compression,
            "index_properties": self.index_properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColoringDatasetConfig':
        return cls(**data)

class ColoringDataset:
    """染色数据集"""
    
    def __init__(self, config: Optional[ColoringDatasetConfig] = None):
        self.config = config or ColoringDatasetConfig()
        self.records: Dict[str, ColoringRecord] = {}
        self.indices: Dict[str, Dict[Any, List[str]]] = defaultdict(dict)
        self._modified = False
        self._save_counter = 0
        
        # 创建索引
        for prop in self.config.index_properties:
            self.indices[prop] = defaultdict(list)
        
        logger.info(f"Coloring dataset initialized: {self.config.name}")
    
    def add_coloring(self, coloring: Coloring, 
                     metadata: ColoringMetadata,
                     coloring_id: Optional[str] = None) -> str:
        """添加染色方案到数据集"""
        if coloring_id is None:
            # 基于染色方案生成ID
            coloring_hash = hashlib.sha256(
                str(coloring.colors).encode() + 
                str(metadata.graph_id).encode()
            ).hexdigest()[:16]
            coloring_id = f"coloring_{coloring_hash}"
        
        if coloring_id in self.records:
            logger.warning(f"Coloring {coloring_id} already exists, updating")
        
        # 创建记录
        record = ColoringRecord(
            coloring_id=coloring_id,
            coloring=coloring,
            metadata=metadata
        )
        
        # 添加到数据集
        self.records[coloring_id] = record
        self._update_indices(coloring_id, coloring, metadata)
        self._modified = True
        
        # 自动保存
        self._auto_save()
        
        logger.debug(f"Coloring added: {coloring_id} for graph {metadata.graph_id}")
        return coloring_id
    
    def _update_indices(self, coloring_id: str, coloring: Coloring, metadata: ColoringMetadata):
        """更新索引"""
        # 更新属性索引
        for prop in self.config.index_properties:
            if prop == "k":
                self.indices[prop][coloring.k].append(coloring_id)
            elif hasattr(metadata, prop):
                value = getattr(metadata, prop)
                if value is not None:
                    self.indices[prop][value].append(coloring_id)
        
        # 更新标签索引
        for tag in metadata.tags:
            if "tags" not in self.indices:
                self.indices["tags"] = defaultdict(list)
            self.indices["tags"][tag].append(coloring_id)
    
    def _auto_save(self):
        """自动保存"""
        if self.config.auto_save:
            self._save_counter += 1
            if self._save_counter >= self.config.save_interval:
                self.save()
                self._save_counter = 0
    
    def get_coloring(self, coloring_id: str) -> Optional[Coloring]:
        """获取染色方案"""
        if coloring_id in self.records:
            return self.records[coloring_id].coloring
        return None
    
    def get_record(self, coloring_id: str) -> Optional[ColoringRecord]:
        """获取染色记录"""
        return self.records.get(coloring_id)
    
    def query(self, **kwargs) -> List[str]:
        """查询染色ID"""
        result_sets = []
        
        for key, value in kwargs.items():
            if key in self.indices:
                if value in self.indices[key]:
                    result_sets.append(set(self.indices[key][value]))
                else:
                    # 如果没有匹配的，返回空结果
                    return []
            elif key == "min_k":
                result_sets.append({
                    cid for cid, record in self.records.items() 
                    if record.coloring.k >= value
                })
            elif key == "max_k":
                result_sets.append({
                    cid for cid, record in self.records.items() 
                    if record.coloring.k <= value
                })
            elif key == "valid_only":
                if value:
                    result_sets.append({
                        cid for cid, record in self.records.items() 
                        if record.coloring.is_valid
                    })
            elif key == "tag":
                if "tags" in self.indices and value in self.indices["tags"]:
                    result_sets.append(set(self.indices["tags"][value]))
                else:
                    return []
        
        # 取交集
        if result_sets:
            result = set.intersection(*result_sets)
            return list(result)
        else:
            # 没有查询条件，返回所有
            return list(self.records.keys())
    
    def get_graph_colorings(self, graph_id: str, 
                           valid_only: bool = False) -> List[ColoringRecord]:
        """获取指定图的所有染色方案"""
        coloring_ids = self.indices["graph_id"].get(graph_id, [])
        
        records = []
        for coloring_id in coloring_ids:
            if coloring_id in self.records:
                record = self.records[coloring_id]
                if not valid_only or record.coloring.is_valid:
                    records.append(record)
        
        return records
    
    def get_best_coloring(self, graph_id: str, 
                         metric: str = "k") -> Optional[ColoringRecord]:
        """获取指定图的最佳染色方案"""
        colorings = self.get_graph_colorings(graph_id, valid_only=True)
        
        if not colorings:
            return None
        
        if metric == "k":
            # 最小颜色数
            return min(colorings, key=lambda r: r.coloring.k)
        elif metric == "score":
            # 最高评分
            return max(colorings, key=lambda r: r.coloring.score)
        elif metric == "runtime":
            # 最快运行时间
            return min(colorings, key=lambda r: r.metadata.runtime)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def update_metadata(self, coloring_id: str, 
                       updates: Dict[str, Any]) -> bool:
        """更新染色元数据"""
        if coloring_id not in self.records:
            return False
        
        record = self.records[coloring_id]
        metadata = record.metadata
        
        # 更新元数据
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
            else:
                metadata.properties[key] = value
        
        record.update()
        self._modified = True
        
        # 重新构建索引
        self._rebuild_indices()
        
        logger.debug(f"Metadata updated for coloring {coloring_id}")
        return True
    
    def _rebuild_indices(self):
        """重建索引"""
        # 清除索引
        for prop in self.indices:
            self.indices[prop].clear()
        
        # 重新构建
        for coloring_id, record in self.records.items():
            self._update_indices(
                coloring_id, 
                record.coloring, 
                record.metadata
            )
    
    def remove_coloring(self, coloring_id: str) -> bool:
        """移除染色方案"""
        if coloring_id in self.records:
            del self.records[coloring_id]
            self._rebuild_indices()
            self._modified = True
            logger.debug(f"Coloring removed: {coloring_id}")
            return True
        return False
    
    def stats(self) -> Dict[str, Any]:
        """数据集统计信息"""
        total_colorings = len(self.records)
        
        if total_colorings == 0:
            return {"total_colorings": 0}
        
        # 颜色数统计
        k_values = [record.coloring.k for record in self.records.values()]
        runtime_values = [record.metadata.runtime for record in self.records.values()]
        score_values = [record.coloring.score for record in self.records.values()]
        
        # 方法分布
        method_dist = defaultdict(int)
        solver_dist = defaultdict(int)
        graph_dist = defaultdict(int)
        tag_dist = defaultdict(int)
        
        for record in self.records.values():
            method_dist[record.metadata.method] += 1
            solver_dist[record.metadata.solver] += 1
            graph_dist[record.metadata.graph_id] += 1
            for tag in record.metadata.tags:
                tag_dist[tag] += 1
        
        # 有效性统计
        valid_count = sum(1 for r in self.records.values() if r.coloring.is_valid)
        
        return {
            "total_colorings": total_colorings,
            "valid_colorings": valid_count,
            "invalid_colorings": total_colorings - valid_count,
            "k_values": {
                "min": min(k_values),
                "max": max(k_values),
                "mean": float(np.mean(k_values)),
                "median": float(np.median(k_values))
            },
            "runtime": {
                "min": min(runtime_values),
                "max": max(runtime_values),
                "mean": float(np.mean(runtime_values)),
                "median": float(np.median(runtime_values))
            },
            "score": {
                "min": min(score_values),
                "max": max(score_values),
                "mean": float(np.mean(score_values)),
                "median": float(np.median(score_values))
            },
            "method_distribution": dict(method_dist),
            "solver_distribution": dict(solver_dist),
            "graphs_with_colorings": len(graph_dist),
            "colorings_per_graph": {
                "min": min(graph_dist.values()),
                "max": max(graph_dist.values()),
                "mean": float(np.mean(list(graph_dist.values()))),
                "median": float(np.median(list(graph_dist.values())))
            },
            "tag_distribution": dict(tag_dist),
            "indexed_properties": list(self.indices.keys())
        }
    
    def save(self, filepath: Optional[str] = None):
        """保存数据集"""
        if filepath is None:
            filepath = f"data/datasets/{self.config.name}.json"
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        data = {
            "config": self.config.to_dict(),
            "records": [record.to_dict() for record in self.records.values()],
            "indices": {
                prop: dict(indices) 
                for prop, indices in self.indices.items()
            }
        }
        
        # 保存
        if self.config.compression:
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        self._modified = False
        logger.info(f"Dataset saved to {filepath} ({len(self.records)} colorings)")
    
    @classmethod
    def load(cls, filepath: str) -> 'ColoringDataset':
        """加载数据集"""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        # 检测是否为压缩格式
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except:
            with open(path, 'r') as f:
                data = json.load(f)
        
        # 创建数据集
        config = ColoringDatasetConfig.from_dict(data["config"])
        dataset = cls(config)
        
        # 加载记录
        for record_data in data["records"]:
            record = ColoringRecord.from_dict(record_data)
            dataset.records[record.coloring_id] = record
        
        # 加载索引
        dataset.indices = defaultdict(dict)
        for prop, indices in data.get("indices", {}).items():
            dataset.indices[prop] = defaultdict(list, indices)
        
        dataset._modified = False
        logger.info(f"Dataset loaded from {filepath} ({len(dataset.records)} colorings)")
        
        return dataset
    
    def export_for_analysis(self, graph_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """导出用于分析的数据"""
        if graph_ids is None:
            graph_ids = list(set(record.metadata.graph_id for record in self.records.values()))
        
        analysis_data = []
        for graph_id in graph_ids:
            colorings = self.get_graph_colorings(graph_id, valid_only=True)
            
            if colorings:
                # 找到最佳染色（最小k）
                best_coloring = min(colorings, key=lambda r: r.coloring.k)
                
                analysis_data.append({
                    "graph_id": graph_id,
                    "best_k": best_coloring.coloring.k,
                    "colorings_count": len(colorings),
                    "k_values": [c.coloring.k for c in colorings],
                    "best_coloring_id": best_coloring.coloring_id,
                    "best_score": best_coloring.coloring.score,
                    "best_runtime": best_coloring.metadata.runtime
                })
        
        return analysis_data
    
    def merge(self, other: 'ColoringDataset') -> 'ColoringDataset':
        """合并另一个数据集"""
        merged_config = ColoringDatasetConfig(
            name=f"{self.config.name}_merged_{other.config.name}",
            description=f"Merged dataset of {self.config.name} and {other.config.name}",
            max_size=self.config.max_size + other.config.max_size,
            auto_save=self.config.auto_save,
            save_interval=self.config.save_interval,
            compression=self.config.compression,
            index_properties=list(set(self.config.index_properties + other.config.index_properties))
        )
        
        merged = ColoringDataset(merged_config)
        
        # 合并记录
        for record in self.records.values():
            merged.records[record.coloring_id] = record.__class__(
                coloring_id=record.coloring_id,
                coloring=Coloring(**record.coloring.to_dict()),
                metadata=ColoringMetadata(**record.metadata.to_dict()),
                created_at=record.created_at,
                updated_at=record.updated_at
            )
        
        for record in other.records.values():
            if record.coloring_id not in merged.records:
                merged.records[record.coloring_id] = record.__class__(
                    coloring_id=record.coloring_id,
                    coloring=Coloring(**record.coloring.to_dict()),
                    metadata=ColoringMetadata(**record.metadata.to_dict()),
                    created_at=record.created_at,
                    updated_at=record.updated_at
                )
        
        # 重建索引
        merged._rebuild_indices()
        
        return merged

# 快捷函数
def load_coloring_dataset(filepath: str) -> ColoringDataset:
    """加载染色数据集的快捷函数"""
    return ColoringDataset.load(filepath)

def save_coloring_dataset(dataset: ColoringDataset, filepath: Optional[str] = None):
    """保存染色数据集的快捷函数"""
    dataset.save(filepath)