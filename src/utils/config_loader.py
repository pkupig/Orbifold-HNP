"""
配置加载器 - 完整实现
支持YAML、JSON配置文件，提供配置验证和转换
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import copy
import os
import re
from datetime import datetime
import warnings

from ..core.base_classes import BaseConfig, Serializable

class ConfigError(Exception):
    """配置错误异常"""
    pass

class ConfigValidator:
    """配置验证器"""
    
    # 配置模式定义
    SCHEMA = {
        "geometry": {
            "type": "dict",
            "required": True,
            "schema": {
                "lattice_type": {"type": "string", "allowed": ["hexagonal", "square", "triangular"]},
                "v1": {"type": "list", "items": [{"type": "float"}, {"type": "float"}]},
                "v2": {"type": "list", "items": [{"type": "float"}, {"type": "float"}]},
                "wrap_to_fundamental": {"type": "boolean", "default": True},
                "periodic_distance_max_shifts": {"type": "integer", "min": 1, "max": 5, "default": 2}
            }
        },
        "graph_builder": {
            "type": "dict",
            "required": True,
            "schema": {
                "epsilon": {"type": "float", "min": 0.0001, "max": 0.1, "default": 0.02},
                "sampling_method": {"type": "string", "allowed": ["fibonacci", "grid", "random", "hybrid"]},
                "use_kdtree": {"type": "boolean", "default": True},
                "search_factor": {"type": "float", "min": 1.0, "max": 3.0, "default": 1.5},
                "min_points": {"type": "integer", "min": 1, "max": 10000, "default": 10},
                "max_points": {"type": "integer", "min": 10, "max": 100000, "default": 1000},
                "jitter": {"type": "float", "min": 0.0, "max": 0.1, "default": 0.0},
                "edge_density_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.1}
            }
        },
        "sat_solver": {
            "type": "dict",
            "required": True,
            "schema": {
                "solver_name": {"type": "string", "allowed": ["kissat", "glucose", "minisat", "cadical"]},
                "timeout": {"type": "integer", "min": 1, "max": 3600, "default": 60},
                "verbose": {"type": "boolean", "default": False},
                "temp_dir": {"type": "string"},
                "keep_temp_files": {"type": "boolean", "default": False},
                "memory_limit": {"type": "string", "regex": r"^\d+[GMK]?$", "default": "4G"}
            }
        },
        "optimizer": {
            "type": "dict",
            "required": True,
            "schema": {
                "method": {"type": "string", "allowed": ["energy_based", "constraint_based", "hybrid", "evolutionary"]},
                "num_candidates": {"type": "integer", "min": 10, "max": 10000, "default": 1000},
                "relaxation_iterations": {"type": "integer", "min": 1, "max": 100, "default": 5},
                "learning_rate": {"type": "float", "min": 0.001, "max": 1.0, "default": 0.1},
                "energy_sigma": {"type": "float", "min": 0.001, "max": 0.1, "default": 0.02},
                "attraction_factor": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.1}
            }
        },
        "pipeline": {
            "type": "dict",
            "required": True,
            "schema": {
                "initial_points": {"type": "integer", "min": 1, "max": 10000, "default": 30},
                "target_k": {"type": "integer", "min": 1, "max": 20, "default": 4},
                "max_iterations": {"type": "integer", "min": 1, "max": 1000, "default": 20},
                "anneal_epsilon": {"type": "boolean", "default": True},
                "epsilon_decay": {"type": "float", "min": 0.1, "max": 0.99, "default": 0.9},
                "min_epsilon": {"type": "float", "min": 0.00001, "max": 0.01, "default": 0.001},
                "save_interval": {"type": "integer", "min": 1, "max": 100, "default": 5},
                "output_dir": {"type": "string", "default": "results"}
            }
        }
    }
    
    @classmethod
    def validate(cls, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证配置
        
        Args:
            config: 配置字典
            
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        # 检查必需的部分
        for section, section_schema in cls.SCHEMA.items():
            if section_schema.get("required", False) and section not in config:
                errors.append(f"Missing required section: {section}")
            elif section in config:
                # 验证子模式
                section_errors = cls._validate_section(
                    config[section], 
                    section_schema.get("schema", {}), 
                    section
                )
                errors.extend(section_errors)
        
        return len(errors) == 0, errors
    
    @classmethod
    def _validate_section(cls, section_data: Any, schema: Dict[str, Any], 
                         path: str = "") -> List[str]:
        """验证配置部分"""
        errors = []
        
        if not isinstance(section_data, dict):
            errors.append(f"{path}: Expected dict, got {type(section_data).__name__}")
            return errors
        
        for key, key_schema in schema.items():
            key_path = f"{path}.{key}" if path else key
            
            if key in section_data:
                value = section_data[key]
                
                # 类型检查
                expected_type = key_schema.get("type")
                if expected_type:
                    type_ok = cls._check_type(value, expected_type, key_schema)
                    if not type_ok:
                        errors.append(
                            f"{key_path}: Expected {expected_type}, "
                            f"got {type(value).__name__}"
                        )
                        continue
                
                # 允许的值检查
                if "allowed" in key_schema:
                    if value not in key_schema["allowed"]:
                        allowed_str = ", ".join(str(v) for v in key_schema["allowed"])
                        errors.append(
                            f"{key_path}: Value {value} not in allowed values: {allowed_str}"
                        )
                
                # 范围检查（数字）
                if isinstance(value, (int, float)):
                    if "min" in key_schema and value < key_schema["min"]:
                        errors.append(
                            f"{key_path}: Value {value} below minimum {key_schema['min']}"
                        )
                    
                    if "max" in key_schema and value > key_schema["max"]:
                        errors.append(
                            f"{key_path}: Value {value} above maximum {key_schema['max']}"
                        )
                
                # 正则表达式检查（字符串）
                if isinstance(value, str) and "regex" in key_schema:
                    if not re.match(key_schema["regex"], value):
                        errors.append(
                            f"{key_path}: Value {value} does not match pattern {key_schema['regex']}"
                        )
            
            elif key_schema.get("required", False):
                errors.append(f"{key_path}: Missing required key")
        
        return errors
    
    @classmethod
    def _check_type(cls, value: Any, expected_type: str, schema: Dict[str, Any]) -> bool:
        """检查类型"""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "float":
            return isinstance(value, (int, float))
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "list":
            if not isinstance(value, list):
                return False
            
            # 检查列表项类型
            if "items" in schema:
                item_schemas = schema["items"]
                if len(item_schemas) != len(value):
                    return False
                
                for i, (item, item_schema) in enumerate(zip(value, item_schemas)):
                    if not cls._check_type(item, item_schema["type"], item_schema):
                        return False
            
            return True
        elif expected_type == "dict":
            return isinstance(value, dict)
        
        return True

def load_config(config_path: Union[str, Path], 
               validate: bool = True) -> Dict[str, Any]:
    """
    从文件加载配置
    
    Args:
        config_path: 配置文件路径
        validate: 是否验证配置
        
    Returns:
        配置字典
    """
    path = Path(config_path)
    
    if not path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    
    # 根据扩展名选择加载器
    if path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ConfigError(f"Unsupported config file format: {path.suffix}")
    
    if config is None:
        config = {}
    
    # 处理环境变量
    config = _resolve_environment_variables(config)
    
    # 处理特殊值
    config = _process_special_values(config)
    
    # 验证配置
    if validate:
        is_valid, errors = ConfigValidator.validate(config)
        if not is_valid:
            error_msg = "Configuration validation failed:\n" + "\n".join(errors)
            raise ConfigError(error_msg)
    
    return config

def save_config(config: Dict[str, Any], config_path: Union[str, Path], 
               format: str = "yaml", **kwargs):
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
        format: 格式 ("yaml" 或 "json")
        **kwargs: 传递给序列化器的额外参数
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 添加元数据
    config_with_meta = copy.deepcopy(config)
    if "_metadata" not in config_with_meta:
        config_with_meta["_metadata"] = {}
    
    config_with_meta["_metadata"].update({
        "saved_at": datetime.now().isoformat(),
        "format": format,
        "version": kwargs.get("version", "1.0.0")
    })
    
    if format == "yaml":
        with open(config_path, 'w') as f:
            yaml.dump(config_with_meta, f, default_flow_style=False, **kwargs)
    elif format == "json":
        with open(config_path, 'w') as f:
            json.dump(config_with_meta, f, indent=2, **kwargs)
    else:
        raise ConfigError(f"Unsupported format: {format}")

def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个配置
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        合并后的配置
    """
    result = copy.deepcopy(base_config)
    
    def recursive_merge(base, override, path=""):
        for key, value in override.items():
            current_path = f"{path}.{key}" if path else key
            
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # 递归合并字典
                recursive_merge(base[key], value, current_path)
            else:
                # 直接覆盖
                base[key] = copy.deepcopy(value)
    
    recursive_merge(result, override_config)
    return result

def load_config_with_overrides(config_path: Union[str, Path], 
                              override_paths: Optional[List[Union[str, Path]]] = None,
                              override_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    加载配置并应用多个覆盖
    
    Args:
        config_path: 主配置文件路径
        override_paths: 覆盖配置文件路径列表
        override_dict: 覆盖配置字典
        
    Returns:
        合并后的配置
    """
    # 加载基础配置
    config = load_config(config_path, validate=False)
    
    # 应用文件覆盖
    if override_paths:
        for override_path in override_paths:
            override_config = load_config(override_path, validate=False)
            config = merge_configs(config, override_config)
    
    # 应用字典覆盖
    if override_dict:
        config = merge_configs(config, override_dict)
    
    # 最终验证
    is_valid, errors = ConfigValidator.validate(config)
    if not is_valid:
        warnings.warn(f"Configuration validation warnings: {errors}")
    
    return config

def generate_config_template(lattice_type: str = "hexagonal", 
                           include_comments: bool = True) -> Dict[str, Any]:
    """
    生成配置模板
    
    Args:
        lattice_type: 晶格类型
        include_comments: 是否包含注释
        
    Returns:
        配置模板字典
    """
    if lattice_type == "hexagonal":
        v1 = [1.0, 0.0]
        v2 = [0.5, 0.8660254037844386]
    elif lattice_type == "square":
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
    elif lattice_type == "triangular":
        v1 = [1.0, 0.0]
        v2 = [0.5, 0.8660254037844386]
    else:
        raise ConfigError(f"Unsupported lattice type: {lattice_type}")
    
    template = {
        "geometry": {
            "lattice_type": lattice_type,
            "v1": v1,
            "v2": v2,
            "wrap_to_fundamental": True,
            "periodic_distance_max_shifts": 2
        },
        "graph_builder": {
            "epsilon": 0.02,
            "sampling_method": "fibonacci" if lattice_type == "hexagonal" else "grid",
            "use_kdtree": True,
            "search_factor": 1.5,
            "min_points": 10,
            "max_points": 1000,
            "jitter": 0.0,
            "edge_density_threshold": 0.1
        },
        "sat_solver": {
            "solver_name": "kissat",
            "timeout": 60,
            "verbose": False,
            "temp_dir": "/tmp",
            "keep_temp_files": False,
            "memory_limit": "4G"
        },
        "optimizer": {
            "method": "constraint_based",
            "num_candidates": 1000,
            "relaxation_iterations": 5,
            "learning_rate": 0.1,
            "energy_sigma": 0.02,
            "attraction_factor": 0.1
        },
        "pipeline": {
            "initial_points": 30,
            "target_k": 4,
            "max_iterations": 20,
            "anneal_epsilon": True,
            "epsilon_decay": 0.9,
            "min_epsilon": 0.001,
            "save_interval": 5,
            "output_dir": "results"
        },
        "visualization": {
            "show_plots": True,
            "save_plots": True,
            "dpi": 300,
            "format": "png"
        }
    }
    
    if include_comments:
        # 添加注释（在YAML中作为字符串存储）
        template["_comments"] = {
            "geometry": "Geometry configuration for the orbifold",
            "graph_builder": "Graph construction parameters",
            "sat_solver": "SAT solver configuration",
            "optimizer": "Graph optimization parameters",
            "pipeline": "Main pipeline configuration",
            "visualization": "Visualization settings"
        }
    
    return template

def diff_configs(config1: Dict[str, Any], 
                config2: Dict[str, Any]) -> Dict[str, Any]:
    """
    比较两个配置的差异
    
    Args:
        config1: 第一个配置
        config2: 第二个配置
        
    Returns:
        差异字典
    """
    def find_diffs(d1, d2, path=""):
        diffs = {}
        
        # 所有键的并集
        all_keys = set(d1.keys()) | set(d2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in d1:
                # 只在d2中存在
                diffs[current_path] = {
                    "type": "added",
                    "value": d2[key]
                }
            elif key not in d2:
                # 只在d1中存在
                diffs[current_path] = {
                    "type": "removed",
                    "value": d1[key]
                }
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                # 递归比较字典
                nested_diffs = find_diffs(d1[key], d2[key], current_path)
                if nested_diffs:
                    diffs.update(nested_diffs)
            elif d1[key] != d2[key]:
                # 值不同
                diffs[current_path] = {
                    "type": "changed",
                    "old_value": d1[key],
                    "new_value": d2[key]
                }
        
        return diffs
    
    return find_diffs(config1, config2)

def print_config_summary(config: Dict[str, Any], 
                        max_depth: int = 2,
                        show_all: bool = False):
    """
    打印配置摘要
    
    Args:
        config: 配置字典
        max_depth: 最大深度
        show_all: 是否显示所有配置项
    """
    def print_section(section, data, depth=0, prefix=""):
        indent = "  " * depth
        
        if isinstance(data, dict):
            print(f"{indent}{prefix}{section}:")
            
            if depth >= max_depth and not show_all:
                print(f"{indent}  ... (truncated)")
                return
            
            for key, value in data.items():
                if key.startswith("_"):
                    continue  # 跳过元数据
                
                if isinstance(value, dict):
                    print_section(key, value, depth + 1)
                elif isinstance(value, list):
                    if len(value) <= 3 or show_all:
                        print(f"{indent}  {key}: {value}")
                    else:
                        print(f"{indent}  {key}: [{value[0]}, {value[1]}, ..., {value[-1]}]")
                else:
                    print(f"{indent}  {key}: {value}")
        else:
            print(f"{indent}{prefix}{section}: {data}")
    
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    
    for section, data in config.items():
        if section.startswith("_"):
            continue  # 跳过元数据
        
        print_section(section, data)

def _resolve_environment_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """解析环境变量"""
    def process_value(value):
        if isinstance(value, str):
            # 替换 ${VAR} 格式的环境变量
            import os
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, value)
            
            for match in matches:
                env_value = os.getenv(match)
                if env_value is not None:
                    value = value.replace(f'${{{match}}}', env_value)
            
            # 替换 $VAR 格式的环境变量
            if value.startswith('$'):
                env_var = value[1:]
                env_value = os.getenv(env_var)
                if env_value is not None:
                    value = env_value
        
        return value
    
    def recursive_resolve(data):
        if isinstance(data, dict):
            return {k: recursive_resolve(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [recursive_resolve(v) for v in data]
        elif isinstance(data, str):
            return process_value(data)
        else:
            return data
    
    return recursive_resolve(config)

def _process_special_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """处理特殊值（如"auto", "now"等）"""
    def process_value(value):
        if isinstance(value, str):
            if value.lower() == "auto":
                # 自动生成值
                return _generate_auto_value()
            elif value.lower() == "now":
                return datetime.now().isoformat()
            elif value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
            elif value.lower() == "null":
                return None
            elif value.isdigit():
                return int(value)
            elif _is_float(value):
                return float(value)
        
        return value
    
    def recursive_process(data):
        if isinstance(data, dict):
            return {k: recursive_process(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [recursive_process(v) for v in data]
        else:
            return process_value(data)
    
    return recursive_process(config)

def _generate_auto_value() -> str:
    """生成自动值"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"auto_{timestamp}"

def _is_float(value: str) -> bool:
    """检查字符串是否可以转换为浮点数"""
    try:
        float(value)
        return True
    except ValueError:
        return False

# 配置管理器类
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: Union[str, Path] = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.configs = {}
    
    def load(self, name: str, validate: bool = True) -> Dict[str, Any]:
        """
        加载配置
        
        Args:
            name: 配置名称（不带扩展名）
            validate: 是否验证
            
        Returns:
            配置字典
        """
        # 尝试多种扩展名
        extensions = ['.yaml', '.yml', '.json']
        
        for ext in extensions:
            config_file = self.config_dir / f"{name}{ext}"
            if config_file.exists():
                config = load_config(config_file, validate=validate)
                self.configs[name] = config
                return config
        
        raise ConfigError(f"No config file found for: {name}")
    
    def save(self, name: str, config: Dict[str, Any], 
            format: str = "yaml"):
        """
        保存配置
        
        Args:
            name: 配置名称
            config: 配置字典
            format: 保存格式
        """
        if format == "yaml":
            ext = ".yaml"
        elif format == "json":
            ext = ".json"
        else:
            raise ConfigError(f"Unsupported format: {format}")
        
        config_file = self.config_dir / f"{name}{ext}"
        save_config(config, config_file, format=format)
        
        # 更新缓存
        self.configs[name] = config
    
    def get(self, name: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """
        获取配置（从缓存）
        
        Args:
            name: 配置名称
            default: 默认值
            
        Returns:
            配置字典或默认值
        """
        return self.configs.get(name, default)
    
    def list_configs(self) -> List[str]:
        """
        列出所有配置
        
        Returns:
            配置名称列表
        """
        configs = []
        
        for ext in ['.yaml', '.yml', '.json']:
            config_files = self.config_dir.glob(f"*{ext}")
            configs.extend([f.stem for f in config_files])
        
        # 去重并排序
        return sorted(set(configs))
    
    def create_template(self, name: str, lattice_type: str = "hexagonal"):
        """
        创建配置模板
        
        Args:
            name: 配置名称
            lattice_type: 晶格类型
        """
        template = generate_config_template(lattice_type, include_comments=True)
        self.save(name, template, format="yaml")
        
        return template