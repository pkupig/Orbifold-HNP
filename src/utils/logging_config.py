"""
日志配置 - 完整实现
提供灵活的日志记录和监控功能
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import colorlog
import time
from datetime import datetime
import json
import threading
from queue import Queue
import atexit

# 自定义日志级别
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, 'SUCCESS')

def success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)

logging.Logger.success = success

# 日志级别颜色映射
LOG_COLORS = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'SUCCESS': 'bold_green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}

# 日志级别背景色
LOG_BG_COLORS = {
    'CRITICAL': 'bg_red',
}

class CustomFormatter(colorlog.ColoredFormatter):
    """自定义日志格式化器"""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        if fmt is None:
            fmt = '%(log_color)s[%(asctime)s] [%(levelname)-8s] [%(name)-20s] %(message)s'
        
        if datefmt is None:
            datefmt = '%Y-%m-%d %H:%M:%S'
        
        super().__init__(
            fmt,
            datefmt=datefmt,
            log_colors=LOG_COLORS,
            secondary_log_colors={
                'message': {
                    'SUCCESS': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red',
                }
            },
            reset=True,
            style='%'
        )
    
    def format(self, record):
        # 添加自定义的SUCCESS级别
        if record.levelno == SUCCESS_LEVEL:
            record.levelname = 'SUCCESS'
        
        return super().format(record)

class FileFormatter(logging.Formatter):
    """文件日志格式化器（无颜色）"""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        if fmt is None:
            fmt = '[%(asctime)s] [%(levelname)-8s] [%(name)-20s] %(message)s'
        
        if datefmt is None:
            datefmt = '%Y-%m-%d %H:%M:%S'
        
        super().__init__(fmt, datefmt)
    
    def format(self, record):
        # 添加自定义的SUCCESS级别
        if record.levelno == SUCCESS_LEVEL:
            record.levelname = 'SUCCESS'
        
        return super().format(record)

class JsonFormatter(logging.Formatter):
    """JSON日志格式化器（用于结构化日志记录）"""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.threadName,
        }
        
        # 添加异常信息
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra'):
            log_obj.update(record.extra)
        
        return json.dumps(log_obj, ensure_ascii=False)

class LogFilter(logging.Filter):
    """日志过滤器"""
    
    def __init__(self, name: str = '', level: Optional[int] = None,
                 modules: Optional[List[str]] = None):
        super().__init__(name)
        self.level = level
        self.modules = modules if modules else []
    
    def filter(self, record):
        # 级别过滤
        if self.level is not None and record.levelno < self.level:
            return False
        
        # 模块过滤
        if self.modules:
            module_match = False
            for module in self.modules:
                if record.name.startswith(module):
                    module_match = True
                    break
            
            if not module_match:
                return False
        
        return True

class AsyncLogHandler(logging.Handler):
    """异步日志处理器"""
    
    def __init__(self, target_handler: logging.Handler):
        super().__init__()
        self.target_handler = target_handler
        self.queue = Queue()
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()
        atexit.register(self.shutdown)
    
    def emit(self, record):
        self.queue.put(record)
    
    def _process_queue(self):
        while True:
            try:
                record = self.queue.get(timeout=1)
                self.target_handler.emit(record)
                self.queue.task_done()
            except:
                continue
    
    def shutdown(self):
        """等待队列处理完成"""
        self.queue.join()

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def record(self, name: str, value: Any, timestamp: Optional[float] = None):
        """记录指标"""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            # 更新当前值
            self.metrics[name] = {
                'value': value,
                'timestamp': timestamp,
                'unit': self._infer_unit(name)
            }
            
            # 添加到历史记录
            self.history.append({
                'name': name,
                'value': value,
                'timestamp': timestamp
            })
    
    def increment(self, name: str, amount: float = 1):
        """递增计数器"""
        current = self.get(name, 0)
        if isinstance(current, (int, float)):
            self.record(name, current + amount)
    
    def get(self, name: str, default: Any = None):
        """获取指标值"""
        with self.lock:
            if name in self.metrics:
                return self.metrics[name]['value']
            return default
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有指标"""
        with self.lock:
            return {k: v['value'] for k, v in self.metrics.items()}
    
    def get_history(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取历史记录"""
        with self.lock:
            if name:
                return [h for h in self.history if h['name'] == name]
            return self.history.copy()
    
    def clear(self):
        """清除指标"""
        with self.lock:
            self.metrics.clear()
            self.history.clear()
    
    def save(self, filepath: str):
        """保存指标到文件"""
        with self.lock:
            data = {
                'metrics': self.metrics,
                'history': self.history,
                'start_time': self.start_time,
                'end_time': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
    
    def _infer_unit(self, name: str) -> str:
        """推断指标单位"""
        name_lower = name.lower()
        
        if any(unit in name_lower for unit in ['time', 'duration', 'runtime']):
            return 'seconds'
        elif any(unit in name_lower for unit in ['memory', 'size', 'bytes']):
            return 'bytes'
        elif any(unit in name_lower for unit in ['count', 'number', 'total']):
            return 'count'
        elif any(unit in name_lower for unit in ['ratio', 'percentage', 'rate']):
            return 'ratio'
        else:
            return 'unitless'

class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total: int, desc: str = "Progress", 
                 logger: Optional[logging.Logger] = None):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        self.logger = logger or get_logger(__name__)
        self.last_update_time = 0
        self.update_interval = 1.0  # 秒
    
    def update(self, n: int = 1):
        """更新进度"""
        self.current += n
        
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self._log_progress()
            self.last_update_time = current_time
    
    def _log_progress(self):
        """记录进度"""
        elapsed = time.time() - self.start_time
        if self.current > 0:
            avg_time_per_item = elapsed / self.current
            remaining_items = self.total - self.current
            eta = avg_time_per_item * remaining_items
            
            progress_pct = (self.current / self.total) * 100
            
            self.logger.info(
                f"{self.desc}: {self.current}/{self.total} "
                f"({progress_pct:.1f}%) - ETA: {eta:.1f}s"
            )
        else:
            self.logger.info(f"{self.desc}: Started")

def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    json_format: bool = False,
    module_levels: Optional[Dict[str, Union[str, int]]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    async_logging: bool = False
) -> None:
    """
    设置日志配置
    
    Args:
        level: 默认日志级别
        log_file: 日志文件路径（可选）
        console: 是否输出到控制台
        json_format: 是否使用JSON格式（文件）
        module_levels: 模块特定日志级别
        log_dir: 日志目录（如果提供，会自动生成日志文件名）
        max_file_size: 日志文件最大大小（字节）
        backup_count: 备份文件数量
        async_logging: 是否使用异步日志
    """
    # 清除现有处理器
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 设置日志级别
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    root_logger.setLevel(level)
    
    handlers = []
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        console_handler.addFilter(LogFilter(level=logging.INFO))  # 控制台只显示INFO及以上
        handlers.append(console_handler)
    
    # 文件处理器
    if log_file is not None or log_dir is not None:
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"orbifold_{timestamp}.log"
        
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用RotatingFileHandler实现日志轮转
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        if json_format:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(FileFormatter())
        
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        handlers.append(file_handler)
    
    # 设置异步日志
    if async_logging:
        async_handlers = []
        for handler in handlers:
            async_handler = AsyncLogHandler(handler)
            async_handlers.append(async_handler)
        
        handlers = async_handlers
    
    # 添加所有处理器
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # 设置模块特定日志级别
    if module_levels:
        for module, module_level in module_levels.items():
            if isinstance(module_level, str):
                module_level = getattr(logging, module_level.upper())
            module_logger = logging.getLogger(module)
            module_logger.setLevel(module_level)
    
    # 禁止第三方库的过多日志
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # 初始日志消息
    root_logger.info(f"Logging initialized at level {logging.getLevelName(level)}")
    if log_file:
        root_logger.info(f"Log file: {log_file}")

def get_logger(name: str, extra: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称
        extra: 额外字段（用于结构化日志）
        
    Returns:
        日志记录器对象
    """
    logger = logging.getLogger(name)
    
    # 添加额外字段
    if extra:
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.extra = extra
            return record
        
        logging.setLogRecordFactory(record_factory)
    
    return logger

class Timer:
    """计时器上下文管理器"""
    
    def __init__(self, name: str = "operation", 
                 logger: Optional[logging.Logger] = None,
                 level: int = logging.INFO):
        self.name = name
        self.logger = logger or get_logger(__name__)
        self.level = level
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.success(f"{self.name} completed in {self.elapsed:.3f} seconds")
        else:
            self.logger.error(f"{self.name} failed after {self.elapsed:.3f} seconds")
    
    def get_elapsed(self) -> Optional[float]:
        """获取已过去的时间"""
        if self.start_time is None:
            return None
        return time.time() - self.start_time

def log_execution_time(level: int = logging.INFO):
    """记录函数执行时间的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            logger.log(level, f"Executing {func.__name__}...")
            result = func(*args, **kwargs)
            
            elapsed = time.time() - start_time
            logger.log(level, f"{func.__name__} executed in {elapsed:.3f} seconds")
            
            return result
        return wrapper
    return decorator

def setup_default_logging():
    """设置默认日志配置"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"orbifold_{timestamp}.log"
    
    setup_logging(
        level="INFO",
        log_file=log_file,
        console=True,
        json_format=False,
        module_levels={
            "src.core": "INFO",
            "src.pipeline": "INFO",
            "src.utils": "INFO",
            "src.visualization": "WARNING"
        }
    )
    
    return str(log_file)

# 全局指标收集器
_global_metrics = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器"""
    return _global_metrics

# 创建默认日志记录器
default_logger = get_logger("orbifold")

def log_success(message: str, *args, **kwargs):
    """记录成功消息"""
    default_logger.success(message, *args, **kwargs)

def log_info(message: str, *args, **kwargs):
    """记录信息消息"""
    default_logger.info(message, *args, **kwargs)

def log_warning(message: str, *args, **kwargs):
    """记录警告消息"""
    default_logger.warning(message, *args, **kwargs)

def log_error(message: str, *args, **kwargs):
    """记录错误消息"""
    default_logger.error(message, *args, **kwargs)

def log_debug(message: str, *args, **kwargs):
    """记录调试消息"""
    default_logger.debug(message, *args, **kwargs)