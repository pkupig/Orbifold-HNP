"""
实验运行器 - 管理批量实验的执行 (已修复 save_graph 调用)
"""
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import concurrent.futures
import time
import traceback
import signal
import sys
import os
from datetime import datetime
import hashlib

from ..core.base_classes import BaseConfig, BaseResult
from ..utils.logging_config import get_logger, setup_logging
# [修复] 导入 save_graph 而不是 save_results
from ..utils.data_io import save_graph, load_results
from .coloring_pipeline import OrbifoldColoringPipeline, PipelineResult

logger = get_logger(__name__)

@dataclass
class ExperimentConfig(BaseConfig):
    """实验配置"""
    
    pipeline_config: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0
    timeout_hours: float = 6.0
    max_memory_gb: float = 8.0
    dependencies: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = super().validate()
        
        if not self.pipeline_config:
            errors.append("Pipeline config cannot be empty")
        
        if self.timeout_hours <= 0:
            errors.append("Timeout must be positive")
        
        if self.max_memory_gb <= 0:
            errors.append("Max memory must be positive")
        
        return errors
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'ExperimentConfig':
        """从YAML文件加载"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls().from_dict(data)
    
    def to_yaml(self, filepath: str):
        """保存为YAML文件"""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

@dataclass
class ExperimentResult(BaseResult):
    """实验结果"""
    
    experiment_id: str = ""
    pipeline_result: Optional[PipelineResult] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = super().to_dict()
        
        if self.pipeline_result:
            result['pipeline_result'] = self.pipeline_result.to_dict()
        
        return result
    
    def from_dict(self, data: Dict[str, Any]) -> 'ExperimentResult':
        """从字典创建"""
        super().from_dict(data)
        
        if 'pipeline_result' in data:
            from .coloring_pipeline import PipelineResult
            self.pipeline_result = PipelineResult().from_dict(data['pipeline_result'])
        
        if 'resource_usage' in data:
            self.resource_usage = data['resource_usage']
        
        if 'artifacts' in data:
            self.artifacts = data['artifacts']
        
        return self

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.start_time = time.time()
        self.peak_memory = 0
        self._initial_memory = self._get_memory_usage()
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0
    
    def update(self):
        """更新资源使用情况"""
        current_memory = self._get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        elapsed = time.time() - self.start_time
        return {
            'elapsed_time_seconds': elapsed,
            'peak_memory_mb': self.peak_memory,
            'current_memory_mb': self._get_memory_usage()
        }

class ExperimentWorker:
    """实验工作器"""
    
    def __init__(self, experiment_id: str, config: ExperimentConfig):
        self.experiment_id = experiment_id
        self.config = config
        self.resource_monitor = ResourceMonitor()
        self.interrupted = False
        
        # 设置中断处理器
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """处理中断信号"""
        self.interrupted = True
        logger.warning(f"Experiment {self.experiment_id} received interrupt signal")
    
    def run(self) -> ExperimentResult:
        """运行实验"""
        result = ExperimentResult(
            experiment_id=self.experiment_id,
            timestamp=datetime.now().isoformat()
        )
        
        try:
            # 检查依赖
            if not self._check_dependencies():
                result.success = False
                result.error_message = "Dependencies not satisfied"
                return result
            
            # 创建管道
            pipeline = OrbifoldColoringPipeline(config_dict=self.config.pipeline_config)
            
            # 运行管道
            start_time = time.time()
            
            def monitor_callback():
                """监控回调函数"""
                if self.interrupted:
                    return False  # 停止管道
                
                # 更新资源使用情况
                self.resource_monitor.update()
                
                # 检查超时
                elapsed = time.time() - start_time
                if elapsed > self.config.timeout_hours * 3600:
                    logger.warning(f"Experiment {self.experiment_id} timeout after {elapsed:.1f}s")
                    return False
                
                # 检查内存
                stats = self.resource_monitor.get_stats()
                if stats['peak_memory_mb'] > self.config.max_memory_gb * 1024:
                    logger.warning(f"Experiment {self.experiment_id} exceeded memory limit")
                    return False
                
                return True  # 继续运行
            
            # 运行管道（带监控）
            pipeline_result = pipeline.run(
                experiment_name=self.experiment_id,
                max_iterations=self.config.parameters.get('max_iterations', 20)
            )
            
            # 收集结果
            result.pipeline_result = pipeline_result
            result.success = pipeline_result.success
            result.execution_time = time.time() - start_time
            result.resource_usage = self.resource_monitor.get_stats()
            
            # 保存产物
            result.artifacts = self._save_artifacts(pipeline)
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Experiment {self.experiment_id} failed: {e}")
            logger.debug(traceback.format_exc())
        
        return result
    
    def _check_dependencies(self) -> bool:
        """检查依赖是否满足"""
        for dep in self.config.dependencies:
            if dep == "kissat":
                try:
                    import subprocess
                    subprocess.run(["kissat", "--version"], 
                                 capture_output=True, check=True)
                except:
                    logger.warning(f"Dependency {dep} not satisfied")
                    return False
        return True
    
    def _save_artifacts(self, pipeline) -> Dict[str, str]:
        """保存实验产物"""
        artifacts = {}
        
        # 保存图
        if pipeline.graph:
            graph_path = f"results/{self.experiment_id}/graph.pkl"
            # [修复] 调用 correct function: save_graph
            save_graph(pipeline.graph, graph_path)
            artifacts['graph'] = graph_path
        
        # 保存可视化
        if hasattr(pipeline, 'visualize_current_state'):
            vis_path = f"results/{self.experiment_id}/visualization.png"
            pipeline.visualize_current_state(save_path=vis_path, show_cover=False)
            artifacts['visualization'] = vis_path
        
        # 保存配置
        config_path = f"results/{self.experiment_id}/config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config.to_dict(), f)
        artifacts['config'] = config_path
        
        return artifacts

class BatchExperiment:
    """批量实验管理器"""
    
    def __init__(self, experiments_dir: str = "experiments", 
                 results_dir: str = "results"):
        self.experiments_dir = Path(experiments_dir)
        self.results_dir = Path(results_dir)
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, ExperimentResult] = {}
        
        # 创建目录
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Batch experiment initialized: {experiments_dir} -> {results_dir}")
    
    def load_experiments(self, pattern: str = "*.yaml"):
        """加载实验配置"""
        config_files = list(self.experiments_dir.glob(pattern))
        
        if not config_files:
            logger.warning(f"No config files found in {self.experiments_dir}")
            return
        
        for config_file in config_files:
            try:
                experiment_id = config_file.stem
                config = ExperimentConfig.from_yaml(str(config_file))
                config.name = experiment_id
                
                # 验证配置
                errors = config.validate()
                if errors:
                    logger.error(f"Invalid config {experiment_id}: {errors}")
                    continue
                
                self.experiments[experiment_id] = config
                logger.info(f"Loaded experiment: {experiment_id}")
                
            except Exception as e:
                logger.error(f"Failed to load {config_file}: {e}")
    
    def run_experiment(self, experiment_id: str, 
                      config: Optional[ExperimentConfig] = None) -> ExperimentResult:
        """运行单个实验"""
        if config is None:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            config = self.experiments[experiment_id]
        
        logger.info(f"Starting experiment: {experiment_id}")
        
        # 创建工作器
        worker = ExperimentWorker(experiment_id, config)
        result = worker.run()
        
        # 保存结果
        self.results[experiment_id] = result
        self._save_result(experiment_id, result)
        
        return result
    
    def run_all(self, max_workers: int = 1, 
               shuffle: bool = False) -> Dict[str, ExperimentResult]:
        """运行所有实验"""
        if not self.experiments:
            self.load_experiments()
        
        if not self.experiments:
            logger.warning("No experiments to run")
            return {}
        
        # 按优先级排序
        experiments_to_run = []
        for exp_id, config in self.experiments.items():
            if config.enabled:
                experiments_to_run.append((exp_id, config))
        
        experiments_to_run.sort(key=lambda x: x[1].priority, reverse=True)
        
        if shuffle:
            import random
            random.shuffle(experiments_to_run)
        
        logger.info(f"Running {len(experiments_to_run)} experiments with {max_workers} workers")
        
        if max_workers <= 1:
            # 顺序执行
            for exp_id, config in experiments_to_run:
                self.run_experiment(exp_id, config)
        else:
            # 并行执行
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 准备任务
                future_to_exp = {}
                for exp_id, config in experiments_to_run:
                    future = executor.submit(self._run_experiment_in_process, exp_id, config.to_dict())
                    future_to_exp[future] = exp_id
                
                # 收集结果
                for future in concurrent.futures.as_completed(future_to_exp):
                    exp_id = future_to_exp[future]
                    try:
                        result_dict = future.result()
                        result = ExperimentResult().from_dict(result_dict)
                        self.results[exp_id] = result
                        self._save_result(exp_id, result)
                        logger.info(f"Completed experiment: {exp_id}")
                    except Exception as e:
                        logger.error(f"Experiment {exp_id} failed: {e}")
        
        return self.results
    
    def _run_experiment_in_process(self, experiment_id: str, 
                                  config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """在独立进程中运行实验"""
        # 在每个进程中重新设置日志
        setup_logging()
        
        config = ExperimentConfig().from_dict(config_dict)
        worker = ExperimentWorker(experiment_id, config)
        result = worker.run()
        
        return result.to_dict()
    
    def _save_result(self, experiment_id: str, result: ExperimentResult):
        """保存实验结果"""
        result_dir = self.results_dir / experiment_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存结果文件
        result_file = result_dir / "result.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.debug(f"Result saved: {result_file}")
    
    def load_results(self):
        """加载已有结果"""
        result_dirs = list(self.results_dir.glob("*"))
        
        for result_dir in result_dirs:
            if result_dir.is_dir():
                result_file = result_dir / "result.json"
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            result_dict = json.load(f)
                        result = ExperimentResult().from_dict(result_dict)
                        self.results[result_dir.name] = result
                    except Exception as e:
                        logger.error(f"Failed to load result {result_dir}: {e}")
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成实验报告"""
        self.load_results()
        
        if not self.results:
            return {"error": "No results available"}
        
        # 统计信息
        total = len(self.results)
        successful = sum(1 for r in self.results.values() if r.success)
        failed = total - successful
        
        # 详细结果
        detailed_results = []
        metrics_summary = {
            'chromatic_numbers': [],
            'execution_times': [],
            'graph_sizes': [],
            'memory_usage': []
        }
        
        for exp_id, result in self.results.items():
            details = {
                'experiment_id': exp_id,
                'success': result.success,
                'execution_time': result.execution_time,
                'error_message': result.error_message
            }
            
            if result.pipeline_result:
                details.update({
                    'chromatic_number': result.pipeline_result.chromatic_number,
                    'graph_size': result.pipeline_result.final_graph_size,
                    'iterations': result.pipeline_result.iterations
                })
                
                # 收集指标
                metrics_summary['chromatic_numbers'].append(
                    result.pipeline_result.chromatic_number
                )
                metrics_summary['execution_times'].append(result.execution_time)
                metrics_summary['graph_sizes'].append(
                    result.pipeline_result.final_graph_size[0]  # 节点数
                )
            
            if result.resource_usage:
                details['resource_usage'] = result.resource_usage
                metrics_summary['memory_usage'].append(
                    result.resource_usage.get('peak_memory_mb', 0)
                )
            
            detailed_results.append(details)
        
        # 计算汇总统计
        import numpy as np
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': total,
            'successful_experiments': successful,
            'failed_experiments': failed,
            'success_rate': successful / total if total > 0 else 0,
            'metrics': {}
        }
        
        for metric_name, values in metrics_summary.items():
            if values:
                summary['metrics'][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        summary['detailed_results'] = detailed_results
        
        # 保存报告
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Report saved to {output_file}")
        
        return summary
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """获取实验状态"""
        if experiment_id in self.results:
            result = self.results[experiment_id]
            return {
                'status': 'completed',
                'success': result.success,
                'execution_time': result.execution_time
            }
        elif experiment_id in self.experiments:
            return {'status': 'pending'}
        else:
            return {'status': 'unknown'}

# 快捷函数
def run_experiment_batch(experiments_dir: str = "experiments",
                        results_dir: str = "results",
                        max_workers: int = 1,
                        report_file: str = "batch_report.json") -> Dict[str, Any]:
    """运行批量实验的快捷函数"""
    batch = BatchExperiment(experiments_dir, results_dir)
    batch.load_experiments()
    batch.run_all(max_workers=max_workers)
    return batch.generate_report(report_file)

def run_experiments_from_directory(directory: str,
                                  pattern: str = "*.yaml",
                                  **kwargs) -> Dict[str, Any]:
    """从目录运行实验的快捷函数"""
    batch = BatchExperiment(directory, directory)
    batch.load_experiments(pattern)
    return batch.run_all(**kwargs)

class ExperimentRunner:
    """实验运行器（单例）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExperimentRunner, cls).__new__(cls)
            cls._instance.batch_experiment = BatchExperiment()
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = ExperimentRunner()
        return cls._instance