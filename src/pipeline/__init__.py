"""
管道模块 - 轨形染色系统的高级工作流管理
"""

from .coloring_pipeline import (
    OrbifoldColoringPipeline,
    PipelineResult,
    run_pipeline
)

from .experiment_runner import (
    ExperimentRunner,
    BatchExperiment,
    ExperimentConfig,
    ExperimentResult,
    run_experiment_batch
)

from .result_handler import (
    ResultHandler,
    analyze_experiment,
    compare_experiments,
    export_results
)

from .monitoring import (
    PipelineMonitor,
    create_monitor
)

__all__ = [
    # 主管道
    'OrbifoldColoringPipeline',
    'PipelineResult',
    'run_pipeline',
    
    # 实验运行器
    'ExperimentRunner',
    'BatchExperiment',
    'ExperimentConfig',
    'ExperimentResult',
    'run_experiment_batch',
    
    # 结果处理器
    'ResultHandler',
    'analyze_experiment',
    'compare_experiments',
    'export_results',
    
    # 监控
    'PipelineMonitor',
    'create_monitor'
]

__version__ = "1.0.0"
__author__ = "Orbifold Coloring System Team"