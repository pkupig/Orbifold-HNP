"""
图数据集 - 完整实现
管理和操作单位距离图的数据集
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import hashlib
from collections import defaultdict
import networkx as nx

from ..core.graph_builder import UnitDistanceGraph, GraphBuilder
from ..core.geometry_engine import GeometryEngine
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class GraphMetadata:
    """图元数据"""
    chromatic_number: Optional[int] = None
    chromatic_upper_bound: Optional[int] = None
    chromatic_lower_bound: Optional[int] = None
    is_unit_distance: bool = True
    lattice_type: str = "hexagonal"
    epsilon: float = 0.02
    generation_method: str = "unknown"
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chromatic_number": self.chromatic_number,
            "chromatic_upper_bound": self.chromatic_upper_bound,
            "chromatic_lower_bound": self.chromatic_lower_bound,
            "is_unit_distance": self.is_unit_distance,
            "lattice_type": self.lattice_type,
            "epsilon": self.epsilon,
            "generation_method": self.generation_method,
            "tags": self.tags,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphMetadata':
        return cls(**data)

@dataclass
class GraphRecord:
    """图记录"""
    graph_id: str
    graph: UnitDistanceGraph
    metadata: GraphMetadata
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
            "graph_id": self.graph_id,
            "graph": {
                "nodes": self.graph.nodes.tolist(),
                "edges": self.graph.edges,
                "epsilon": self.graph.epsilon
            },
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphRecord':
        graph_data = data["graph"]
        graph = UnitDistanceGraph(
            nodes=np.array(graph_data["nodes"]),
            edges=[tuple(e) for e in graph_data["edges"]],
            epsilon=graph_data["epsilon"]
        )
        
        metadata = GraphMetadata.from_dict(data["metadata"])
        
        return cls(
            graph_id=data["graph_id"],
            graph=graph,
            metadata=metadata,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", "")
        )

@dataclass
class GraphDatasetConfig:
    """图数据集配置"""
    name: str = "graph_dataset"
    description: str = ""
    max_size: int = 1000
    auto_save: bool = True
    save_interval: int = 10
    compression: bool = True
    index_properties: List[str] = field(default_factory=lambda: ["chromatic_number", "lattice_type"])
    
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
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphDatasetConfig':
        return cls(**data)

class GraphDataset:
    """图数据集"""
    
    def __init__(self, config: Optional[GraphDatasetConfig] = None):
        self.config = config or GraphDatasetConfig()
        self.records: Dict[str, GraphRecord] = {}
        self.indices: Dict[str, Dict[Any, List[str]]] = defaultdict(dict)
        self._modified = False
        self._save_counter = 0
        
        # 创建索引
        for prop in self.config.index_properties:
            self.indices[prop] = defaultdict(list)
        
        logger.info(f"Graph dataset initialized: {self.config.name}")
    
    def add_graph(self, graph: UnitDistanceGraph, 
                  metadata: Optional[GraphMetadata] = None,
                  graph_id: Optional[str] = None) -> str:
        """添加图到数据集"""
        if graph_id is None:
            # 基于图内容生成ID
            graph_hash = hashlib.sha256(
                graph.nodes.tobytes() + 
                str(graph.edges).encode()
            ).hexdigest()[:16]
            graph_id = f"graph_{graph_hash}"
        
        if graph_id in self.records:
            logger.warning(f"Graph {graph_id} already exists, updating")
        
        # 创建元数据
        if metadata is None:
            metadata = GraphMetadata(
                lattice_type="unknown",
                epsilon=graph.epsilon
            )
        
        # 创建记录
        record = GraphRecord(
            graph_id=graph_id,
            graph=graph,
            metadata=metadata
        )
        
        # 添加到数据集
        self.records[graph_id] = record
        self._update_indices(graph_id, metadata)
        self._modified = True
        
        # 自动保存
        self._auto_save()
        
        logger.debug(f"Graph added: {graph_id} ({graph.num_nodes} nodes)")
        return graph_id
    
    def _update_indices(self, graph_id: str, metadata: GraphMetadata):
        """更新索引"""
        # 更新属性索引
        for prop in self.config.index_properties:
            if hasattr(metadata, prop):
                value = getattr(metadata, prop)
                if value is not None:
                    self.indices[prop][value].append(graph_id)
        
        # 更新标签索引
        for tag in metadata.tags:
            if "tags" not in self.indices:
                self.indices["tags"] = defaultdict(list)
            self.indices["tags"][tag].append(graph_id)
    
    def _auto_save(self):
        """自动保存"""
        if self.config.auto_save:
            self._save_counter += 1
            if self._save_counter >= self.config.save_interval:
                self.save()
                self._save_counter = 0
    
    def get_graph(self, graph_id: str) -> Optional[UnitDistanceGraph]:
        """获取图"""
        if graph_id in self.records:
            return self.records[graph_id].graph
        return None
    
    def get_record(self, graph_id: str) -> Optional[GraphRecord]:
        """获取图记录"""
        return self.records.get(graph_id)
    
    def query(self, **kwargs) -> List[str]:
        """查询图ID"""
        result_sets = []
        
        for key, value in kwargs.items():
            if key in self.indices:
                if value in self.indices[key]:
                    result_sets.append(set(self.indices[key][value]))
                else:
                    # 如果没有匹配的，返回空结果
                    return []
            elif key == "min_nodes":
                result_sets.append({
                    gid for gid, record in self.records.items() 
                    if record.graph.num_nodes >= value
                })
            elif key == "max_nodes":
                result_sets.append({
                    gid for gid, record in self.records.items() 
                    if record.graph.num_nodes <= value
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
    
    def filter(self, condition: callable) -> List[str]:
        """使用条件函数过滤图"""
        result = []
        
        for graph_id, record in self.records.items():
            if condition(record):
                result.append(graph_id)
        
        return result
    
    def update_metadata(self, graph_id: str, 
                       updates: Dict[str, Any]) -> bool:
        """更新图元数据"""
        if graph_id not in self.records:
            return False
        
        record = self.records[graph_id]
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
        
        logger.debug(f"Metadata updated for graph {graph_id}")
        return True
    
    def _rebuild_indices(self):
        """重建索引"""
        # 清除索引
        for prop in self.indices:
            self.indices[prop].clear()
        
        # 重新构建
        for graph_id, record in self.records.items():
            self._update_indices(graph_id, record.metadata)
    
    def remove_graph(self, graph_id: str) -> bool:
        """移除图"""
        if graph_id in self.records:
            del self.records[graph_id]
            self._rebuild_indices()
            self._modified = True
            logger.debug(f"Graph removed: {graph_id}")
            return True
        return False
    
    def stats(self) -> Dict[str, Any]:
        """数据集统计信息"""
        total_graphs = len(self.records)
        
        if total_graphs == 0:
            return {"total_graphs": 0}
        
        # 节点数统计
        node_counts = [record.graph.num_nodes for record in self.records.values()]
        edge_counts = [record.graph.num_edges for record in self.records.values()]
        
        # 色数统计
        chromatic_numbers = []
        for record in self.records.values():
            if record.metadata.chromatic_number is not None:
                chromatic_numbers.append(record.metadata.chromatic_number)
        
        # 晶格类型分布
        lattice_dist = defaultdict(int)
        tag_dist = defaultdict(int)
        
        for record in self.records.values():
            lattice_dist[record.metadata.lattice_type] += 1
            for tag in record.metadata.tags:
                tag_dist[tag] += 1
        
        return {
            "total_graphs": total_graphs,
            "nodes": {
                "min": min(node_counts),
                "max": max(node_counts),
                "mean": float(np.mean(node_counts)),
                "median": float(np.median(node_counts))
            },
            "edges": {
                "min": min(edge_counts),
                "max": max(edge_counts),
                "mean": float(np.mean(edge_counts)),
                "median": float(np.median(edge_counts))
            },
            "chromatic_numbers": {
                "min": min(chromatic_numbers) if chromatic_numbers else None,
                "max": max(chromatic_numbers) if chromatic_numbers else None,
                "distribution": dict(sorted(
                    [(k, v) for k, v in defaultdict(int, np.bincount(chromatic_numbers)).items()],
                    key=lambda x: x[0]
                )) if chromatic_numbers else {}
            },
            "lattice_distribution": dict(lattice_dist),
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
        logger.info(f"Dataset saved to {filepath} ({len(self.records)} graphs)")
    
    @classmethod
    def load(cls, filepath: str) -> 'GraphDataset':
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
        config = GraphDatasetConfig.from_dict(data["config"])
        dataset = cls(config)
        
        # 加载记录
        for record_data in data["records"]:
            record = GraphRecord.from_dict(record_data)
            dataset.records[record.graph_id] = record
        
        # 加载索引
        dataset.indices = defaultdict(dict)
        for prop, indices in data.get("indices", {}).items():
            dataset.indices[prop] = defaultdict(list, indices)
        
        dataset._modified = False
        logger.info(f"Dataset loaded from {filepath} ({len(dataset.records)} graphs)")
        
        return dataset
    
    def export_to_networkx(self, graph_ids: Optional[List[str]] = None) -> List[nx.Graph]:
        """导出为NetworkX图"""
        if graph_ids is None:
            graph_ids = list(self.records.keys())
        
        nx_graphs = []
        for graph_id in graph_ids:
            if graph_id in self.records:
                nx_graph = self.records[graph_id].graph.to_networkx()
                # 添加元数据作为图属性
                nx_graph.graph.update(self.records[graph_id].metadata.to_dict())
                nx_graph.graph["graph_id"] = graph_id
                nx_graphs.append(nx_graph)
        
        return nx_graphs
    
    def split(self, ratios: List[float], 
             seed: Optional[int] = None) -> List['GraphDataset']:
        """分割数据集"""
        if abs(sum(ratios) - 1.0) > 1e-10:
            raise ValueError("Ratios must sum to 1.0")
        
        # 获取所有图ID并打乱
        all_ids = list(self.records.keys())
        if seed is not None:
            import random
            random.seed(seed)
            random.shuffle(all_ids)
        
        # 计算分割点
        n = len(all_ids)
        split_points = [0]
        cumulative = 0
        for ratio in ratios:
            cumulative += ratio
            split_points.append(int(cumulative * n))
        
        # 创建子数据集
        datasets = []
        for i in range(len(ratios)):
            start = split_points[i]
            end = split_points[i + 1]
            subset_ids = all_ids[start:end]
            
            # 创建新数据集
            subset_config = GraphDatasetConfig(
                name=f"{self.config.name}_split_{i}",
                description=f"Split {i} of {self.config.name}",
                max_size=self.config.max_size,
                auto_save=self.config.auto_save,
                save_interval=self.config.save_interval,
                compression=self.config.compression,
                index_properties=self.config.index_properties
            )
            
            subset = GraphDataset(subset_config)
            
            for graph_id in subset_ids:
                subset.records[graph_id] = self.records[graph_id].__class__(
                    graph_id=graph_id,
                    graph=UnitDistanceGraph(
                        nodes=self.records[graph_id].graph.nodes.copy(),
                        edges=self.records[graph_id].graph.edges.copy(),
                        epsilon=self.records[graph_id].graph.epsilon
                    ),
                    metadata=GraphMetadata(**self.records[graph_id].metadata.to_dict()),
                    created_at=self.records[graph_id].created_at,
                    updated_at=self.records[graph_id].updated_at
                )
            
            # 重建索引
            subset._rebuild_indices()
            datasets.append(subset)
        
        return datasets

# 快捷函数
def load_graph_dataset(filepath: str) -> GraphDataset:
    """加载图数据集的快捷函数"""
    return GraphDataset.load(filepath)

def save_graph_dataset(dataset: GraphDataset, filepath: Optional[str] = None):
    """保存图数据集的快捷函数"""
    dataset.save(filepath)