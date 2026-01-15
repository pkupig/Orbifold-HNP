"""
数据输入输出工具 - 完整实现
支持图的保存、加载、转换和批量处理
"""
import json
import pickle
import yaml
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import zipfile
import tempfile
import shutil
import h5py
import pandas as pd
from datetime import datetime

from ..core.graph_builder import UnitDistanceGraph, GraphBuilder
from ..core.base_classes import Vector2D, BoundingBox

def save_graph(graph: UnitDistanceGraph, filepath: str, format: str = "pkl", 
              compress: bool = False) -> str:
    """
    保存图到文件，支持多种格式
    
    Args:
        graph: 单位距离图
        filepath: 文件路径
        format: 格式 ("pkl", "json", "npz", "hdf5", "graphml")
        compress: 是否压缩
        
    Returns:
        保存的文件路径
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "pkl":
        data = {
            "nodes": graph.nodes,
            "edges": graph.edges,
            "epsilon": graph.epsilon,
            "metadata": graph.metadata
        }
        
        mode = 'wb'
        if compress:
            import gzip
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    elif format == "json":
        data = {
            "nodes": graph.nodes.tolist(),
            "edges": graph.edges,
            "epsilon": graph.epsilon,
            "metadata": graph.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=_json_serializer)
    
    elif format == "npz":
        np.savez_compressed(
            filepath,
            nodes=graph.nodes,
            edges=np.array(graph.edges, dtype=np.int32),
            epsilon=np.array([graph.epsilon]),
            metadata=json.dumps(graph.metadata)
        )
    
    elif format == "hdf5":
        with h5py.File(filepath, 'w') as f:
            # 保存节点
            nodes_ds = f.create_dataset("nodes", data=graph.nodes)
            nodes_ds.attrs["description"] = "Node coordinates (x, y)"
            
            # 保存边
            edges_ds = f.create_dataset("edges", data=np.array(graph.edges, dtype=np.int32))
            edges_ds.attrs["description"] = "Edge list (u, v)"
            
            # 保存元数据
            f.attrs["epsilon"] = graph.epsilon
            for key, value in graph.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    f.attrs[f"metadata_{key}"] = value
                else:
                    f.attrs[f"metadata_{key}"] = str(value)
    
    elif format == "graphml":
        nx_graph = graph.to_networkx()
        nx.write_graphml(nx_graph, filepath)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return filepath

def load_graph(filepath: str, format: Optional[str] = None) -> UnitDistanceGraph:
    """
    从文件加载图
    
    Args:
        filepath: 文件路径
        format: 格式 (自动检测如果为None)
        
    Returns:
        单位距离图
    """
    path = Path(filepath)
    
    if format is None:
        # 从扩展名推断格式
        if path.suffix == ".pkl" or path.suffix == ".pickle":
            format = "pkl"
        elif path.suffix == ".gz" and path.stem.endswith(".pkl"):
            format = "pkl"
            filepath = str(path)
        elif path.suffix == ".json":
            format = "json"
        elif path.suffix == ".npz":
            format = "npz"
        elif path.suffix == ".h5" or path.suffix == ".hdf5":
            format = "hdf5"
        elif path.suffix == ".graphml":
            format = "graphml"
        else:
            # 默认使用pickle
            format = "pkl"
    
    if format == "pkl":
        # 检查是否压缩
        if str(filepath).endswith('.gz'):
            import gzip
            with gzip.open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        
        return UnitDistanceGraph(
            nodes=data["nodes"],
            edges=[tuple(edge) for edge in data["edges"]],
            epsilon=data["epsilon"],
            metadata=data.get("metadata", {})
        )
    
    elif format == "json":
        with open(filepath, 'r') as f:
            data = json.load(f, object_hook=_json_deserializer)
        
        return UnitDistanceGraph(
            nodes=np.array(data["nodes"]),
            edges=[tuple(edge) for edge in data["edges"]],
            epsilon=data["epsilon"],
            metadata=data.get("metadata", {})
        )
    
    elif format == "npz":
        data = np.load(filepath, allow_pickle=True)
        
        nodes = data["nodes"]
        edges = [tuple(edge) for edge in data["edges"]]
        epsilon = float(data["epsilon"][0])
        
        # 加载元数据
        metadata_str = data["metadata"].item() if isinstance(data["metadata"], np.ndarray) else data["metadata"]
        metadata = json.loads(metadata_str) if metadata_str else {}
        
        return UnitDistanceGraph(
            nodes=nodes,
            edges=edges,
            epsilon=epsilon,
            metadata=metadata
        )
    
    elif format == "hdf5":
        with h5py.File(filepath, 'r') as f:
            nodes = f["nodes"][:]
            edges = [tuple(edge) for edge in f["edges"][:]]
            epsilon = f.attrs["epsilon"]
            
            # 加载元数据
            metadata = {}
            for key in f.attrs:
                if key.startswith("metadata_"):
                    metadata_key = key[9:]  # 去掉"metadata_"前缀
                    metadata[metadata_key] = f.attrs[key]
            
            return UnitDistanceGraph(
                nodes=nodes,
                edges=edges,
                epsilon=epsilon,
                metadata=metadata
            )
    
    elif format == "graphml":
        nx_graph = nx.read_graphml(filepath)
        
        # 从NetworkX图提取数据
        nodes = []
        node_index_map = {}
        
        for i, (node_id, node_data) in enumerate(nx_graph.nodes(data=True)):
            if "pos" in node_data:
                pos = node_data["pos"]
                if isinstance(pos, str):
                    # 解析字符串格式的位置
                    pos = eval(pos)
                nodes.append(pos)
            else:
                nodes.append([0, 0])  # 默认位置
            node_index_map[node_id] = i
        
        # 提取边
        edges = []
        for u, v in nx_graph.edges():
            u_idx = node_index_map[u]
            v_idx = node_index_map[v]
            edges.append((u_idx, v_idx))
        
        # 从图属性获取epsilon
        epsilon = float(nx_graph.graph.get("epsilon", 0.02))
        
        # 从图属性获取元数据
        metadata = {k: v for k, v in nx_graph.graph.items() if k != "epsilon"}
        
        return UnitDistanceGraph(
            nodes=np.array(nodes),
            edges=edges,
            epsilon=epsilon,
            metadata=metadata
        )
    
    else:
        raise ValueError(f"Unsupported format: {format}")

def save_coloring(coloring: Dict[int, int], filepath: str):
    """
    保存染色方案
    
    Args:
        coloring: 染色方案 {节点索引: 颜色索引}
        filepath: 文件路径
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "coloring": coloring,
        "timestamp": datetime.now().isoformat(),
        "num_colors": len(set(coloring.values()))
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_coloring(filepath: str) -> Dict[int, int]:
    """
    加载染色方案
    
    Args:
        filepath: 文件路径
        
    Returns:
        染色方案字典
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # 将键转换回整数（JSON保存时键会被转换为字符串）
    coloring = {int(k): v for k, v in data["coloring"].items()}
    return coloring

def save_results(results: Dict[str, Any], directory: str, 
                experiment_name: str, format: str = "json") -> str:
    """
    保存实验结果
    
    Args:
        results: 实验结果字典
        directory: 目录路径
        experiment_name: 实验名称
        format: 保存格式
        
    Returns:
        保存的目录路径
    """
    exp_dir = Path(directory) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存结果摘要
    if format == "json":
        result_file = exp_dir / "results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=_json_serializer)
    elif format == "yaml":
        result_file = exp_dir / "results.yaml"
        with open(result_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    
    # 如果结果中包含图，单独保存
    if "graph" in results and isinstance(results["graph"], UnitDistanceGraph):
        save_graph(results["graph"], str(exp_dir / "graph.pkl"))
    
    # 如果结果中包含染色方案，单独保存
    if "coloring" in results and isinstance(results["coloring"], dict):
        save_coloring(results["coloring"], str(exp_dir / "coloring.json"))
    
    # 保存元数据
    metadata = {
        "experiment_name": experiment_name,
        "saved_at": datetime.now().isoformat(),
        "format": format
    }
    
    with open(exp_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return str(exp_dir)

def load_results(directory: str, experiment_name: str, 
                load_graph: bool = True) -> Dict[str, Any]:
    """
    加载实验结果
    
    Args:
        directory: 目录路径
        experiment_name: 实验名称
        load_graph: 是否加载图
        
    Returns:
        结果字典
    """
    exp_dir = Path(directory) / experiment_name
    
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    results = {}
    
    # 加载结果文件（尝试多种格式）
    result_files = list(exp_dir.glob("results.*"))
    if result_files:
        result_file = result_files[0]
        suffix = result_file.suffix.lower()
        
        if suffix == ".json":
            with open(result_file, 'r') as f:
                results.update(json.load(f, object_hook=_json_deserializer))
        elif suffix in [".yaml", ".yml"]:
            with open(result_file, 'r') as f:
                results.update(yaml.safe_load(f))
    
    # 加载图
    if load_graph:
        graph_files = list(exp_dir.glob("graph.*"))
        if graph_files:
            try:
                graph = load_graph(str(graph_files[0]))
                results["graph"] = graph
            except Exception as e:
                print(f"Warning: Failed to load graph: {e}")
    
    # 加载染色方案
    coloring_file = exp_dir / "coloring.json"
    if coloring_file.exists():
        try:
            coloring = load_coloring(str(coloring_file))
            results["coloring"] = coloring
        except Exception as e:
            print(f"Warning: Failed to load coloring: {e}")
    
    # 加载元数据
    metadata_file = exp_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            results["_metadata"] = json.load(f)
    
    results["experiment_name"] = experiment_name
    results["experiment_dir"] = str(exp_dir)
    
    return results

def export_to_csv(data: Dict[str, Any], filepath: str, 
                 include_graph_stats: bool = False):
    """
    导出数据到CSV
    
    Args:
        data: 数据字典
        filepath: 输出文件路径
        include_graph_stats: 是否包含图统计信息
    """
    df_data = []
    
    if isinstance(data, dict) and "results" in data:
        # 批量结果
        for result in data["results"]:
            row = _extract_result_row(result, include_graph_stats)
            df_data.append(row)
    elif isinstance(data, list):
        # 结果列表
        for result in data:
            row = _extract_result_row(result, include_graph_stats)
            df_data.append(row)
    elif isinstance(data, dict):
        # 单个结果
        row = _extract_result_row(data, include_graph_stats)
        df_data.append(row)
    
    if df_data:
        df = pd.DataFrame(df_data)
        df.to_csv(filepath, index=False)
        print(f"Exported {len(df_data)} records to {filepath}")
    else:
        print("No data to export")

def _extract_result_row(result: Dict[str, Any], include_graph_stats: bool) -> Dict[str, Any]:
    """从结果中提取行数据"""
    row = {}
    
    # 基础信息
    row["experiment_name"] = result.get("experiment_name", "")
    row["success"] = result.get("success", False)
    row["chromatic_number"] = result.get("chromatic_number", -1)
    row["runtime"] = result.get("runtime", 0.0)
    
    # 图大小
    graph_size = result.get("final_graph_size", (0, 0))
    if isinstance(graph_size, (list, tuple)) and len(graph_size) >= 2:
        row["nodes"] = graph_size[0]
        row["edges"] = graph_size[1]
    
    # 迭代信息
    row["iterations"] = result.get("iterations", 0)
    
    # 配置信息
    config = result.get("config", {})
    if "geometry" in config:
        row["lattice_type"] = config["geometry"].get("lattice_type", "")
    if "pipeline" in config:
        row["target_k"] = config["pipeline"].get("target_k", 0)
    
    # 图统计信息
    if include_graph_stats and "graph" in result:
        graph = result["graph"]
        row["epsilon"] = graph.epsilon
        row["edge_density"] = graph.edge_density
        row["avg_degree"] = np.mean(graph.get_node_degrees()) if graph.num_nodes > 0 else 0
    
    return row

def create_experiment_archive(experiment_name: str, 
                            source_dir: str = "results",
                            output_file: Optional[str] = None) -> str:
    """
    创建实验存档（ZIP文件）
    
    Args:
        experiment_name: 实验名称
        source_dir: 源目录
        output_file: 输出文件路径（可选）
        
    Returns:
        存档文件路径
    """
    exp_dir = Path(source_dir) / experiment_name
    
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{experiment_name}_{timestamp}.zip"
    
    # 创建ZIP文件
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in exp_dir.rglob("*"):
            if file_path.is_file():
                # 在ZIP中创建相对路径
                arcname = file_path.relative_to(exp_dir.parent)
                zipf.write(file_path, arcname)
    
    # 添加元数据
    with zipfile.ZipFile(output_file, 'a') as zipf:
        metadata = {
            "experiment_name": experiment_name,
            "created_at": datetime.now().isoformat(),
            "source_directory": str(exp_dir),
            "total_files": len(list(exp_dir.rglob("*")))
        }
        
        metadata_str = json.dumps(metadata, indent=2)
        zipf.writestr("_metadata.json", metadata_str)
    
    return output_file

def extract_experiment_archive(zip_file: str, 
                              target_dir: str = "results",
                              overwrite: bool = False) -> str:
    """
    提取实验存档
    
    Args:
        zip_file: ZIP文件路径
        target_dir: 目标目录
        overwrite: 是否覆盖现有文件
        
    Returns:
        提取的目录路径
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_file, 'r') as zipf:
        # 获取实验名称（从元数据或第一个文件推断）
        experiment_name = None
        
        # 尝试从元数据获取
        if "_metadata.json" in zipf.namelist():
            metadata_str = zipf.read("_metadata.json").decode()
            metadata = json.loads(metadata_str)
            experiment_name = metadata.get("experiment_name")
        
        # 如果无法从元数据获取，从文件名推断
        if experiment_name is None:
            # 获取第一个目录
            first_file = zipf.namelist()[0]
            experiment_name = Path(first_file).parts[0]
        
        exp_dir = target_path / experiment_name
        
        if exp_dir.exists() and not overwrite:
            # 创建唯一目录
            counter = 1
            while exp_dir.exists():
                new_name = f"{experiment_name}_{counter}"
                exp_dir = target_path / new_name
                counter += 1
        
        # 提取文件
        zipf.extractall(target_dir)
    
    return str(exp_dir)

def convert_graph_format(input_file: str, output_file: str, 
                        output_format: str = "json") -> str:
    """
    转换图文件格式
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        output_format: 输出格式
        
    Returns:
        输出文件路径
    """
    # 加载图
    graph = load_graph(input_file)
    
    # 保存为新格式
    save_graph(graph, output_file, format=output_format)
    
    return output_file

def batch_process_graphs(input_dir: str, output_dir: str, 
                        process_func: callable, pattern: str = "*.pkl",
                        **kwargs) -> List[str]:
    """
    批量处理图文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        process_func: 处理函数，接受UnitDistanceGraph和kwargs，返回处理后的图
        pattern: 文件匹配模式
        **kwargs: 传递给处理函数的参数
        
    Returns:
        处理后的文件路径列表
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    
    for input_file in input_path.glob(pattern):
        try:
            # 加载图
            graph = load_graph(str(input_file))
            
            # 处理图
            processed_graph = process_func(graph, **kwargs)
            
            # 保存处理后的图
            output_file = output_path / input_file.name
            save_graph(processed_graph, str(output_file))
            
            processed_files.append(str(output_file))
            
            print(f"Processed: {input_file.name} -> {output_file.name}")
            
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
    
    return processed_files

def merge_graphs(graph_files: List[str], output_file: str) -> UnitDistanceGraph:
    """
    合并多个图文件
    
    Args:
        graph_files: 图文件路径列表
        output_file: 输出文件路径
        
    Returns:
        合并后的图
    """
    if not graph_files:
        raise ValueError("No graph files provided")
    
    # 加载第一个图
    merged_graph = load_graph(graph_files[0])
    
    # 合并其余图
    for graph_file in graph_files[1:]:
        graph = load_graph(graph_file)
        
        # 检查epsilon是否一致
        if abs(merged_graph.epsilon - graph.epsilon) > 1e-10:
            print(f"Warning: Epsilon mismatch between graphs: "
                  f"{merged_graph.epsilon} vs {graph.epsilon}")
        
        # 合并节点和边
        merged_nodes = np.vstack([merged_graph.nodes, graph.nodes])
        
        # 调整第二个图的边索引
        offset = len(merged_graph.nodes)
        adjusted_edges = [(u + offset, v + offset) for u, v in graph.edges]
        
        # 合并边
        merged_edges = merged_graph.edges + adjusted_edges
        
        # 合并元数据
        merged_metadata = {**merged_graph.metadata, **graph.metadata}
        merged_metadata["merged_from"] = merged_metadata.get("merged_from", []) + [graph_file]
        
        merged_graph = UnitDistanceGraph(
            nodes=merged_nodes,
            edges=merged_edges,
            epsilon=merged_graph.epsilon,  # 使用第一个图的epsilon
            metadata=merged_metadata
        )
    
    # 保存合并后的图
    save_graph(merged_graph, output_file)
    
    return merged_graph

def _json_serializer(obj):
    """JSON序列化辅助函数"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def _json_deserializer(obj):
    """JSON反序列化辅助函数"""
    if isinstance(obj, dict):
        # 检查是否应该转换为Vector2D或BoundingBox
        if 'x' in obj and 'y' in obj and len(obj) == 2:
            return Vector2D(obj['x'], obj['y'])
        elif all(k in obj for k in ['min_x', 'min_y', 'max_x', 'max_y']):
            return BoundingBox(**{k: obj[k] for k in ['min_x', 'min_y', 'max_x', 'max_y']})
    return obj