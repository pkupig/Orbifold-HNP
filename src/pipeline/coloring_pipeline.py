"""
染色管道 - 完整实现 (已修复保存和序列化问题)
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import yaml
import json
import time
import pickle
from pathlib import Path
import warnings

from ..core.geometry_engine import GeometryEngine, LatticeConfig
from ..core.graph_builder import GraphBuilder, GraphConfig, UnitDistanceGraph
from ..core.sat_solver import SATSolver, SATConfig
from ..core.optimizer import GraphOptimizer, OptimizerConfig
from ..visualization.graph_visualizer import GraphVisualizer
from ..utils.logging_config import setup_logging, get_logger
# 引入专门的保存函数
from ..utils.data_io import save_graph

logger = get_logger(__name__)

@dataclass
class PipelineConfig:
    """管道配置类"""
    initial_points: int = 30
    target_k: int = 4
    max_iterations: int = 20
    anneal_epsilon: bool = True
    epsilon_decay: float = 0.9
    min_epsilon: float = 0.001
    save_interval: int = 5
    output_dir: str = "results"
    checkpoint_interval: int = 10
    max_runtime_hours: float = 12.0
    early_stopping_patience: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_points": self.initial_points,
            "target_k": self.target_k,
            "max_iterations": self.max_iterations,
            "anneal_epsilon": self.anneal_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "save_interval": self.save_interval,
            "output_dir": self.output_dir,
            "checkpoint_interval": self.checkpoint_interval,
            "max_runtime_hours": self.max_runtime_hours,
            "early_stopping_patience": self.early_stopping_patience
        }

@dataclass
class PipelineState:
    """管道状态类"""
    graph: Optional[UnitDistanceGraph] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_iteration: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    result: Optional['PipelineResult'] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_graph": self.graph is not None,
            "history_length": len(self.history),
            "current_iteration": self.current_iteration,
            "config_keys": list(self.config.keys()),
            "has_result": self.result is not None,
            "start_time": self.start_time
        }

@dataclass
class PipelineResult:
    """管道运行结果"""
    # [修复] 添加默认值以支持无参初始化
    success: bool = False
    chromatic_number: int = 0
    final_graph_size: Tuple[int, int] = (0, 0)
    iterations: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    runtime: float = 0.0
    termination_reason: str = "unknown"
    graph: Optional[UnitDistanceGraph] = None
    coloring: Optional[Dict[int, int]] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "chromatic_number": self.chromatic_number,
            "final_graph_size": list(self.final_graph_size),
            "iterations": self.iterations,
            "runtime": self.runtime,
            "termination_reason": self.termination_reason,
            "config_summary": {
                "lattice_type": self.config.get("geometry", {}).get("lattice_type"),
                "target_k": self.config.get("pipeline", {}).get("target_k"),
                "epsilon": self.config.get("graph_builder", {}).get("epsilon")
            },
            "has_graph": self.graph is not None,
            "has_coloring": self.coloring is not None,
            "stats": self.stats,
            # 保存完整信息以便恢复
            "history": self.history,
            "config": self.config
        }

    def from_dict(self, data: Dict[str, Any]) -> 'PipelineResult':
        """[修复] 从字典加载数据的辅助方法"""
        if "success" in data: self.success = data["success"]
        if "chromatic_number" in data: self.chromatic_number = data["chromatic_number"]
        if "final_graph_size" in data: self.final_graph_size = tuple(data["final_graph_size"])
        if "iterations" in data: self.iterations = data["iterations"]
        if "runtime" in data: self.runtime = data["runtime"]
        if "termination_reason" in data: self.termination_reason = data["termination_reason"]
        if "history" in data: self.history = data["history"]
        if "config" in data: self.config = data["config"]
        if "stats" in data: self.stats = data["stats"]
        return self
    
    def save(self, filepath: str):
        """保存结果到文件"""
        data = self.to_dict()
        
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        # 如果需要，单独保存图
        if self.graph is not None:
            graph_path = Path(filepath).parent / "graph.pkl"
            # [修复] 使用 data_io.save_graph 替代错误的实例方法调用
            try:
                save_graph(self.graph, str(graph_path))
            except Exception as e:
                logger.error(f"Failed to save graph using data_io: {e}. Falling back to pickle.")
                # 备选方案：直接 Pickle
                with open(graph_path, 'wb') as f:
                    pickle.dump({
                        'nodes': self.graph.nodes,
                        'edges': self.graph.edges,
                        'epsilon': self.graph.epsilon,
                        'metadata': self.graph.metadata
                    }, f)
            
        logger.info(f"Result saved to {filepath}")

class OrbifoldColoringPipeline:
    """轨形染色管道"""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """初始化管道"""
        self.config = self._load_config(config_path, config_dict)
        self._initialize_modules()
        
        # 状态变量
        self.graph = None
        self.history = []
        self.current_iteration = 0
        self.result = None
        self.start_time = None
        
        logger.info(f"Pipeline initialized with config from {config_path or 'dict'}")
    
    def _load_config(self, config_path: Optional[str], config_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """加载配置"""
        if config_dict is not None:
            return config_dict
        
        if config_path is None:
            # 使用默认配置
            default_path = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
            config_path = str(default_path)
            if not Path(config_path).exists():
                 logger.warning(f"Default config not found at {config_path}, using empty config")
                 return {}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    def _initialize_modules(self):
        """初始化所有模块"""
        # 几何引擎
        geo_config_dict = self.config.get("geometry", {})
        geo_config = LatticeConfig(
            type=geo_config_dict.get("lattice_type", "hexagonal"),
            v1=tuple(geo_config_dict.get("v1", [1.0, 0.0])),
            v2=tuple(geo_config_dict.get("v2", [0.5, 0.86602540378])),
            wrap_to_fundamental=geo_config_dict.get("wrap_to_fundamental", True)
        )
        self.geometry = GeometryEngine(geo_config)
        
        # 图构建器
        graph_config_dict = self.config.get("graph_builder", {})
        graph_config = GraphConfig(
            epsilon=graph_config_dict.get("epsilon", 0.02),
            sampling_method=graph_config_dict.get("sampling_method", "fibonacci"),
            use_kdtree=graph_config_dict.get("use_kdtree", True),
            search_factor=graph_config_dict.get("search_factor", 1.5),
            min_points=graph_config_dict.get("min_points", 10),
            max_points=graph_config_dict.get("max_points", 1000),
            jitter=graph_config_dict.get("jitter", 0.0)
        )
        self.builder = GraphBuilder(self.geometry, graph_config)
        
        # SAT求解器
        sat_config_dict = self.config.get("sat_solver", {})
        sat_config = SATConfig(
            solver_name=sat_config_dict.get("solver_name", "kissat"),
            timeout=sat_config_dict.get("timeout", 60),
            verbose=sat_config_dict.get("verbose", False),
            temp_dir=sat_config_dict.get("temp_dir", "/tmp"),
            keep_temp_files=sat_config_dict.get("keep_temp_files", False)
        )
        self.solver = SATSolver(sat_config)
        
        # 优化器
        opt_config_dict = self.config.get("optimizer", {})
        opt_config = OptimizerConfig(
            method=opt_config_dict.get("method", "constraint_based"),
            num_candidates=opt_config_dict.get("num_candidates", 1000),
            relaxation_iterations=opt_config_dict.get("relaxation_iterations", 5),
            learning_rate=opt_config_dict.get("learning_rate", 0.1),
            energy_sigma=opt_config_dict.get("energy_sigma", 0.02),
            attraction_factor=opt_config_dict.get("attraction_factor", 0.1),
            max_energy_evaluations=opt_config_dict.get("max_energy_evaluations", 1000)
        )
        self.optimizer = GraphOptimizer(self.geometry, self.builder, opt_config)
        
        # 可视化工具
        self.visualizer = GraphVisualizer()
        
        logger.info("All modules initialized successfully")
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
        
        deep_update(self.config, updates)
        self._initialize_modules()
        logger.info("Configuration updated")
    
    def initialize_graph(self, method: Optional[str] = None, n_points: Optional[int] = None):
        """初始化图"""
        pipeline_config = self.config.get("pipeline", {})
        
        n_points = n_points or pipeline_config.get("initial_points", 30)
        method = method or self.config.get("graph_builder", {}).get("sampling_method", "fibonacci")
        
        logger.info(f"Initializing graph with {n_points} points using {method} sampling")
        
        # 生成点
        nodes = self.builder.initialize_points(n_points, method=method)
        
        # 构建图
        self.graph = self.builder.construct_graph(nodes)
        
        # 记录历史
        self.history.append({
            "action": "initialize",
            "iteration": 0,
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "epsilon": self.graph.epsilon,
            "method": method
        })
        
        logger.info(f"Graph initialized: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
    
    def run_iteration(self) -> Tuple[bool, Optional[Dict[int, int]], str]:
        """运行一次迭代"""
        pipeline_config = self.config.get("pipeline", {})
        target_k = pipeline_config.get("target_k", 4)
        
        logger.debug(f"Iteration {self.current_iteration}: Testing {target_k}-colorability")
        
        # 测试当前图是否可染色
        colorable, coloring, stats = self.solver.is_k_colorable(self.graph, target_k)
        
        if colorable is None:  # 超时或错误
            message = f"SAT solver timeout or error in iteration {self.current_iteration}"
            logger.warning(message)
            return None, None, message
        
        if not colorable:  # 找到反例！
            message = f"Found non-{target_k}-colorable graph in iteration {self.current_iteration}"
            logger.success(message)
            return False, None, message
        
        # 图可染色，寻找难以染色的点
        logger.debug(f"Graph is {target_k}-colorable, finding hard point...")
        new_point = self.optimizer.find_hard_point(self.graph, coloring)
        
        # 添加新点
        self.graph = self.builder.add_node_with_edges(self.graph, new_point)
        
        # 退火epsilon
        if pipeline_config.get("anneal_epsilon", True):
            old_epsilon = self.graph.epsilon
            self.graph = self.optimizer.anneal_epsilon(
                self.graph,
                decay_factor=pipeline_config.get("epsilon_decay", 0.9),
                min_epsilon=pipeline_config.get("min_epsilon", 0.001)
            )
            if abs(self.graph.epsilon - old_epsilon) > 1e-10:
                logger.debug(f"Epsilon annealed from {old_epsilon:.4f} to {self.graph.epsilon:.4f}")
        
        # 定期松弛节点
        if self.current_iteration % 3 == 0:  # 每3次迭代松弛一次
            self.graph = self.optimizer.relax_nodes(self.graph)
        
        # 记录历史
        self.history.append({
            "action": "add_point",
            "iteration": self.current_iteration,
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "epsilon": self.graph.epsilon,
            "point": new_point.tolist(),
            "colorable": True
        })
        
        return True, coloring, f"Added point {new_point.tolist()}"
    
    def estimate_chromatic_number(self, max_k: int = 8) -> Tuple[int, Dict[str, Any]]:
        """估计当前图的色数"""
        logger.info(f"Estimating chromatic number (max k={max_k})")
        
        chromatic_num, stats = self.solver.estimate_chromatic_number(
            self.graph, 
            max_k=max_k,
            timeout_per_test=min(30, self.config.get("sat_solver", {}).get("timeout", 60) // 2)
        )
        
        return chromatic_num, stats
    
    def run(self, experiment_name: str = "experiment", 
            max_iterations: Optional[int] = None) -> PipelineResult:
        """运行完整管道"""
        self.start_time = time.time()
        pipeline_config = self.config.get("pipeline", {})
        
        max_iterations = max_iterations or pipeline_config.get("max_iterations", 20)
        save_interval = pipeline_config.get("save_interval", 5)
        output_dir = Path(pipeline_config.get("output_dir", "results")) / experiment_name
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting pipeline for experiment: {experiment_name}")
        logger.info(f"Max iterations: {max_iterations}, Target k: {pipeline_config.get('target_k', 4)}")
        
        # 1. 初始化图
        if self.graph is None:
            self.initialize_graph()
        
        # 2. 迭代循环
        termination_reason = "max_iterations_reached"
        
        for iteration in range(1, max_iterations + 1):
            self.current_iteration = iteration
            logger.info(f"\n--- Iteration {iteration}/{max_iterations} ---")
            logger.info(f"Graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            
            # 运行一次迭代
            status, coloring, message = self.run_iteration()
            
            if status is None:  # 错误/超时
                termination_reason = "solver_error"
                break
            
            if status is False:  # 找到反例
                termination_reason = "counterexample_found"
                break
            
            # 定期保存状态
            if save_interval > 0 and iteration % save_interval == 0:
                self.save_state(output_dir / f"iteration_{iteration}.pkl")
                logger.info(f"Checkpoint saved at iteration {iteration}")
        
        # 3. 估计色数
        runtime = time.time() - self.start_time
        chromatic_num, stats = self.estimate_chromatic_number()
        
        # 4. 创建结果对象
        self.result = PipelineResult(
            success=(termination_reason == "counterexample_found"),
            chromatic_number=chromatic_num,
            final_graph_size=(len(self.graph.nodes), len(self.graph.edges)),
            iterations=self.current_iteration,
            history=self.history,
            config=self.config,
            runtime=runtime,
            termination_reason=termination_reason,
            graph=self.graph,
            stats=stats
        )
        
        # 5. 保存结果
        self.result.save(output_dir / "result.json")
        
        # 6. 可视化
        if self.config.get("visualization", {}).get("save_plots", True):
            self.visualize_current_state(
                save_path=output_dir / "graph.png",
                show_cover=False
            )
        
        logger.info(f"\n{'='*60}")
        logger.info("Pipeline completed!")
        logger.info(f"Runtime: {runtime:.2f} seconds")
        logger.info(f"Iterations: {self.current_iteration}")
        logger.info(f"Final graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        logger.info(f"Chromatic number: {chromatic_num}")
        logger.info(f"Success: {self.result.success}")
        logger.info(f"Termination reason: {termination_reason}")
        logger.info('='*60)
        
        return self.result
    
    def save_state(self, filepath: str):
        """保存当前状态到文件"""
        state = {
            'graph': self.graph,
            'history': self.history,
            'current_iteration': self.current_iteration,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.debug(f"State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """从文件加载状态"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.graph = state['graph']
        self.history = state['history']
        self.current_iteration = state['current_iteration']
        self.config = state['config']
        
        # 重新初始化模块
        self._initialize_modules()
        
        logger.info(f"State loaded from {filepath}")
    
    def visualize_current_state(self, save_path: Optional[str] = None, 
                              show_cover: bool = False):
        """可视化当前状态"""
        if self.graph is None:
            logger.warning("No graph to visualize")
            return
        
        vis_config = self.config.get("visualization", {})
        
        if show_cover:
            fig = self.visualizer.plot_cover(
                self.graph, 
                self.geometry,
                title=f"Cover Space (Iteration {self.current_iteration})"
            )
        else:
            # 如果有染色方案，显示染色
            coloring = None
            if self.result and self.result.coloring:
                coloring = self.result.coloring
            
            fig = self.visualizer.plot_graph(
                self.graph, 
                self.geometry,
                coloring=coloring,
                title=f"Unit Distance Graph (Iteration {self.current_iteration})"
            )
        
        if save_path is not None and vis_config.get("save_plots", True):
            dpi = vis_config.get("dpi", 300)
            # 确保目录存在
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        if vis_config.get("show_plots", True):
            import matplotlib.pyplot as plt
            plt.show()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取管道统计信息"""
        if self.result is None:
            return {"status": "not_run"}
        
        stats = {
            "runtime": self.result.runtime,
            "iterations": self.result.iterations,
            "success": self.result.success,
            "chromatic_number": self.result.chromatic_number,
            "final_graph": {
                "nodes": len(self.graph.nodes) if self.graph else 0,
                "edges": len(self.graph.edges) if self.graph else 0
            },
            "termination_reason": self.result.termination_reason,
            "solver_stats": self.solver.get_statistics(),
            "optimizer_history": self.optimizer.get_history_summary()
        }
        
        return stats

# 快捷函数
def run_pipeline(config_path: str, experiment_name: str = "experiment") -> PipelineResult:
    """运行管道的快捷函数"""
    pipeline = OrbifoldColoringPipeline(config_path)
    return pipeline.run(experiment_name)