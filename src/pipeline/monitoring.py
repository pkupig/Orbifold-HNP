"""
管道监控模块 - 完整实现
监控管道运行状态，收集指标，跟踪进度
"""
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class Metric:
    """指标数据类"""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "tags": self.tags
        }

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics: List[Metric] = []
        self._lock = threading.Lock()
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """记录指标"""
        with self._lock:
            metric = Metric(
                name=name,
                value=value,
                tags=tags or {}
            )
            self.metrics.append(metric)
    
    def record_batch(self, metrics: Dict[str, float], tags: Optional[Dict[str, str]] = None):
        """批量记录指标"""
        with self._lock:
            for name, value in metrics.items():
                self.metrics.append(
                    Metric(name=name, value=value, tags=tags or {})
                )
    
    def get_metrics(self, name: Optional[str] = None, 
                   tags: Optional[Dict[str, str]] = None) -> List[Metric]:
        """获取指标"""
        with self._lock:
            filtered = self.metrics
            
            if name is not None:
                filtered = [m for m in filtered if m.name == name]
            
            if tags is not None:
                for key, value in tags.items():
                    filtered = [m for m in filtered if m.tags.get(key) == value]
            
            return filtered.copy()
    
    def get_metric_values(self, name: str, 
                         tags: Optional[Dict[str, str]] = None) -> List[float]:
        """获取指标值列表"""
        metrics = self.get_metrics(name, tags)
        return [m.value for m in metrics]
    
    def get_statistics(self, name: str, 
                      tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """获取指标的统计信息"""
        values = self.get_metric_values(name, tags)
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75))
        }
    
    def clear(self):
        """清除所有指标"""
        with self._lock:
            self.metrics.clear()
    
    def to_dataframe(self):
        """转换为pandas DataFrame（如果可用）"""
        try:
            import pandas as pd
            
            with self._lock:
                data = [m.to_dict() for m in self.metrics]
            
            df = pd.DataFrame(data)
            
            # 展开tags列
            if not df.empty and df["tags"].iloc[0]:
                tags_df = pd.json_normalize(df["tags"])
                df = pd.concat([df.drop("tags", axis=1), tags_df], axis=1)
            
            return df
            
        except ImportError:
            logger.warning("pandas not available, cannot create DataFrame")
            return None

class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total: int, description: str = ""):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        # 阶段跟踪
        self.phases: Dict[str, Dict[str, Any]] = {}
        self.current_phase = None
    
    def update(self, n: int = 1, status: Optional[str] = None):
        """更新进度"""
        with self._lock:
            self.current = min(self.current + n, self.total)
        
        if status:
            logger.info(status)
    
    def set_phase(self, phase_name: str, total_steps: Optional[int] = None):
        """设置当前阶段"""
        with self._lock:
            self.current_phase = phase_name
            
            if phase_name not in self.phases:
                self.phases[phase_name] = {
                    "start_time": time.time(),
                    "total_steps": total_steps,
                    "current_step": 0,
                    "completed": False
                }
    
    def update_phase(self, n: int = 1):
        """更新阶段进度"""
        if self.current_phase is None:
            return
        
        with self._lock:
            phase = self.phases[self.current_phase]
            if phase["total_steps"] is not None:
                phase["current_step"] = min(phase["current_step"] + n, phase["total_steps"])
    
    def complete_phase(self):
        """标记阶段完成"""
        if self.current_phase is None:
            return
        
        with self._lock:
            phase = self.phases[self.current_phase]
            phase["completed"] = True
            phase["end_time"] = time.time()
            phase["duration"] = phase["end_time"] - phase["start_time"]
        
        self.current_phase = None
    
    @property
    def progress(self) -> float:
        """当前进度（0到1之间）"""
        if self.total == 0:
            return 0.0
        return self.current / self.total
    
    @property
    def elapsed(self) -> float:
        """已过去的时间（秒）"""
        return time.time() - self.start_time
    
    @property
    def eta(self) -> Optional[float]:
        """估计剩余时间（秒）"""
        if self.progress == 0:
            return None
        return self.elapsed / self.progress * (1 - self.progress)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取进度摘要"""
        with self._lock:
            phase_summary = {}
            for phase_name, phase_data in self.phases.items():
                if "end_time" in phase_data:
                    phase_summary[phase_name] = {
                        "duration": phase_data["duration"],
                        "completed": phase_data["completed"]
                    }
                else:
                    phase_summary[phase_name] = {
                        "current_step": phase_data["current_step"],
                        "total_steps": phase_data["total_steps"],
                        "completed": phase_data["completed"]
                    }
            
            return {
                "description": self.description,
                "total": self.total,
                "current": self.current,
                "progress": self.progress,
                "elapsed": self.elapsed,
                "eta": self.eta,
                "phases": phase_summary
            }
    
    def format_progress_bar(self, width: int = 40) -> str:
        """格式化进度条"""
        filled = int(width * self.progress)
        bar = "█" * filled + "░" * (width - filled)
        
        percent = f"{self.progress * 100:6.2f}%"
        elapsed_str = str(timedelta(seconds=int(self.elapsed)))
        
        if self.eta is not None:
            eta_str = str(timedelta(seconds=int(self.eta)))
            return f"{self.description} |{bar}| {percent} [{elapsed_str}<{eta_str}, {self.current}/{self.total}]"
        else:
            return f"{self.description} |{bar}| {percent} [{elapsed_str}, {self.current}/{self.total}]"

class PipelineMonitor:
    """管道监控器"""
    
    def __init__(self, pipeline_name: str = "orbifold_pipeline"):
        self.pipeline_name = pipeline_name
        self.metrics_collector = MetricsCollector()
        self.progress_trackers: Dict[str, ProgressTracker] = {}
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # 性能监控
        self.performance_stats = defaultdict(list)
        
        logger.info(f"Pipeline monitor initialized: {pipeline_name}")
    
    def create_progress_tracker(self, name: str, total: int, 
                               description: str = "") -> ProgressTracker:
        """创建进度跟踪器"""
        with self._lock:
            tracker = ProgressTracker(total, description)
            self.progress_trackers[name] = tracker
            return tracker
    
    def get_progress_tracker(self, name: str) -> Optional[ProgressTracker]:
        """获取进度跟踪器"""
        return self.progress_trackers.get(name)
    
    def record_event(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """记录事件"""
        with self._lock:
            event = {
                "type": event_type,
                "timestamp": time.time(),
                "data": data or {}
            }
            self.events.append(event)
        
        logger.debug(f"Event recorded: {event_type}")
    
    def record_performance(self, operation: str, duration: float):
        """记录性能数据"""
        self.performance_stats[operation].append(duration)
        self.metrics_collector.record(f"performance.{operation}", duration)
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """获取性能摘要"""
        summary = {}
        
        for operation, durations in self.performance_stats.items():
            if durations:
                summary[operation] = {
                    "count": len(durations),
                    "total": sum(durations),
                    "mean": np.mean(durations),
                    "std": np.std(durations),
                    "min": min(durations),
                    "max": max(durations)
                }
        
        return summary
    
    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告"""
        with self._lock:
            # 进度报告
            progress_report = {}
            for name, tracker in self.progress_trackers.items():
                progress_report[name] = tracker.get_summary()
            
            # 事件计数
            event_counts = defaultdict(int)
            for event in self.events:
                event_counts[event["type"]] += 1
            
            return {
                "pipeline_name": self.pipeline_name,
                "timestamp": datetime.now().isoformat(),
                "progress": progress_report,
                "event_counts": dict(event_counts),
                "performance_summary": self.get_performance_summary(),
                "total_metrics_recorded": len(self.metrics_collector.metrics)
            }
    
    def print_status(self):
        """打印当前状态"""
        report = self.get_status_report()
        
        print(f"\n{'='*60}")
        print(f"Pipeline Monitor: {self.pipeline_name}")
        print(f"Time: {report['timestamp']}")
        print('='*60)
        
        # 打印进度
        for name, progress in report['progress'].items():
            tracker = self.progress_trackers[name]
            print(f"\n{name}: {tracker.format_progress_bar()}")
        
        # 打印性能摘要
        if report['performance_summary']:
            print(f"\nPerformance Summary:")
            for operation, stats in report['performance_summary'].items():
                print(f"  {operation}: {stats['mean']:.3f}s avg "
                      f"({stats['count']} calls, {stats['total']:.1f}s total)")
        
        # 打印事件
        if report['event_counts']:
            print(f"\nEvents:")
            for event_type, count in report['event_counts'].items():
                print(f"  {event_type}: {count}")
        
        print('='*60)
    
    def save_report(self, filepath: str):
        """保存报告到文件"""
        import json
        
        report = self.get_status_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Monitor report saved to {filepath}")
    
    def clear(self):
        """清除所有监控数据"""
        with self._lock:
            self.metrics_collector.clear()
            self.progress_trackers.clear()
            self.events.clear()
            self.performance_stats.clear()

# 快捷函数
def create_monitor(pipeline_name: str = "orbifold_pipeline") -> PipelineMonitor:
    """创建管道监控器"""
    return PipelineMonitor(pipeline_name)

def get_default_metrics() -> List[str]:
    """获取默认监控指标列表"""
    return [
        "graph.nodes",
        "graph.edges",
        "graph.density",
        "pipeline.iteration",
        "pipeline.epsilon",
        "solver.time",
        "solver.result",
        "optimizer.energy",
        "optimizer.candidates"
    ]