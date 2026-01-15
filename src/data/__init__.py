"""
数据模块 - 轨形染色系统的数据管理
"""

from .graph_dataset import (
    GraphDataset,
    GraphDatasetConfig,
    load_graph_dataset,
    save_graph_dataset
)

from .coloring_dataset import (
    ColoringDataset,
    ColoringDatasetConfig,
    load_coloring_dataset,
    save_coloring_dataset
)

__all__ = [
    # 图数据集
    'GraphDataset',
    'GraphDatasetConfig',
    'load_graph_dataset',
    'save_graph_dataset',
    
    # 染色数据集
    'ColoringDataset',
    'ColoringDatasetConfig',
    'load_coloring_dataset',
    'save_coloring_dataset'
]

__version__ = "1.0.0"
__author__ = "Orbifold Coloring System Team"