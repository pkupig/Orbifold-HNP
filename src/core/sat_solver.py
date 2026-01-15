"""
SAT求解器模块 - 完整的实现
将图染色问题转化为SAT问题并求解
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
import subprocess
import tempfile
import os
import time
import shutil
import warnings

from .graph_builder import UnitDistanceGraph

@dataclass
class SATConfig:
    """SAT求解器配置"""
    solver_name: str = "kissat"  # kissat, glucose, minisat, cadical
    timeout: int = 60  # 超时时间(秒)
    verbose: bool = False
    temp_dir: str = "/tmp"
    keep_temp_files: bool = False
    memory_limit: str = "4G"  # 内存限制
    
    def __post_init__(self):
        """验证参数"""
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)

class SATSolver:
    """
    SAT求解器：将图染色问题转化为SAT问题并求解
    """
    
    def __init__(self, config: Optional[SATConfig] = None):
        """
        初始化SAT求解器
        
        Args:
            config: SAT求解器配置
        """
        self.config = config or SATConfig()
        self._check_solver_available()
        
        # 统计信息
        self.stats = {
            "total_solves": 0,
            "satisfiable": 0,
            "unsatisfiable": 0,
            "timeouts": 0,
            "total_time": 0.0
        }
    
    def _check_solver_available(self):
        """检查求解器是否可用"""
        solver_path = self._find_solver_path()
        if solver_path is None:
            raise RuntimeError(
                f"SAT solver '{self.config.solver_name}' not found. "
                f"Please install it (e.g., 'apt-get install {self.config.solver_name}' "
                f"or compile from source)."
            )
        self.solver_path = solver_path
    
    def _find_solver_path(self) -> Optional[str]:
        """查找求解器路径"""
        # 首先尝试直接调用
        try:
            result = subprocess.run(
                [self.config.solver_name, "--version"],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0 or "version" in result.stdout.lower():
                return self.config.solver_name
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # 尝试在常见位置查找
        common_paths = [
            f"/usr/bin/{self.config.solver_name}",
            f"/usr/local/bin/{self.config.solver_name}",
            f"/opt/homebrew/bin/{self.config.solver_name}",
            f"./{self.config.solver_name}",
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        return None
    
    def graph_to_cnf(self, graph: UnitDistanceGraph, k: int, 
                     encoding: str = "standard") -> str:
        """
        将图k染色问题转换为DIMACS CNF格式
        
        Args:
            graph: 单位距离图
            k: 颜色数
            encoding: 编码方式 ("standard", "efficient", "log")
            
        Returns:
            DIMACS格式的CNF字符串
        """
        n_nodes = graph.num_nodes
        
        if encoding == "log":
            # 对数编码（变量更少但子句更多）
            return self._graph_to_cnf_log_encoding(graph, k)
        elif encoding == "efficient":
            # 高效编码（添加冗余约束）
            return self._graph_to_cnf_efficient(graph, k)
        else:
            # 标准编码
            return self._graph_to_cnf_standard(graph, k)
    
    def _graph_to_cnf_standard(self, graph: UnitDistanceGraph, k: int) -> str:
        """标准CNF编码"""
        n_nodes = graph.num_nodes
        
        # 变量映射: node i 颜色 c 对应变量 (i * k + c + 1)
        def var_id(node_idx: int, color: int) -> int:
            return node_idx * k + color + 1
        
        # 收集子句
        clauses = []
        
        # 约束1: 每个节点至少有一种颜色
        for i in range(n_nodes):
            clause = [var_id(i, c) for c in range(k)]
            clauses.append(clause)
        
        # 约束2: 每个节点最多有一种颜色
        for i in range(n_nodes):
            for c1 in range(k):
                for c2 in range(c1 + 1, k):
                    clause = [-var_id(i, c1), -var_id(i, c2)]
                    clauses.append(clause)
        
        # 约束3: 相邻节点不能同色
        for u, v in graph.edges:
            for c in range(k):
                clause = [-var_id(u, c), -var_id(v, c)]
                clauses.append(clause)
        
        # 构建DIMACS格式
        n_vars = n_nodes * k
        n_clauses = len(clauses)
        
        dimacs = f"p cnf {n_vars} {n_clauses}\n"
        for clause in clauses:
            dimacs += " ".join(str(lit) for lit in clause) + " 0\n"
        
        return dimacs
    
    def _graph_to_cnf_efficient(self, graph: UnitDistanceGraph, k: int) -> str:
        """高效CNF编码（添加对称性破缺等优化）"""
        n_nodes = graph.num_nodes
        
        # 变量映射
        def var_id(node_idx: int, color: int) -> int:
            return node_idx * k + color + 1
        
        clauses = []
        
        # 约束1: 每个节点至少有一种颜色
        for i in range(n_nodes):
            clause = [var_id(i, c) for c in range(k)]
            clauses.append(clause)
        
        # 约束2: 每个节点最多有一种颜色（使用顺序编码减少子句数）
        for i in range(n_nodes):
            # 使用顺序编码（sequential encoding）
            # 引入辅助变量表示"节点i的颜色 <= c"
            aux_vars = {}
            aux_start = n_nodes * k + 1
            
            # 创建辅助变量
            for c in range(k-1):
                aux_vars[(i, c)] = aux_start
                aux_start += 1
            
            # 顺序编码约束
            # 第一个颜色
            clauses.append([-var_id(i, 0), aux_vars[(i, 0)]])
            
            # 中间颜色
            for c in range(1, k-1):
                clauses.append([-var_id(i, c), aux_vars[(i, c)]])
                clauses.append([-aux_vars[(i, c-1)], aux_vars[(i, c)]])
                clauses.append([-var_id(i, c), -aux_vars[(i, c-1)]])
            
            # 最后一个颜色
            if k > 1:
                clauses.append([-var_id(i, k-1), -aux_vars[(i, k-2)]])
        
        # 约束3: 相邻节点不能同色
        for u, v in graph.edges:
            for c in range(k):
                clause = [-var_id(u, c), -var_id(v, c)]
                clauses.append(clause)
        
        # 对称性破缺：固定第一个节点的颜色
        if n_nodes > 0 and k > 0:
            clauses.append([var_id(0, 0)])  # 节点0必须为颜色0
        
        # 更新变量数量（包括辅助变量）
        n_vars = aux_start - 1 if k > 1 else n_nodes * k
        n_clauses = len(clauses)
        
        dimacs = f"p cnf {n_vars} {n_clauses}\n"
        for clause in clauses:
            dimacs += " ".join(str(lit) for lit in clause) + " 0\n"
        
        return dimacs
    
    def _graph_to_cnf_log_encoding(self, graph: UnitDistanceGraph, k: int) -> str:
        """对数编码（使用二进制表示颜色）"""
        n_nodes = graph.num_nodes
        
        # 计算需要的位数
        bits = max(1, int(np.ceil(np.log2(k))))
        
        # 变量映射: 节点i的第b位对应变量 (i * bits + b + 1)
        def var_id(node_idx: int, bit: int) -> int:
            return node_idx * bits + bit + 1
        
        clauses = []
        
        # 约束: 禁止无效的颜色编码（大于等于k的二进制数）
        invalid_encodings = []
        for code in range(2**bits):
            if code >= k:
                # 这个编码无效，需要禁止
                invalid_encodings.append(code)
        
        for code in invalid_encodings:
            for i in range(n_nodes):
                clause = []
                for b in range(bits):
                    bit_value = (code >> b) & 1
                    literal = var_id(i, b) if bit_value == 1 else -var_id(i, b)
                    clause.append(literal)
                clauses.append(clause)
        
        # 约束: 相邻节点不能同色
        for u, v in graph.edges:
            # 对于所有可能的颜色
            for color in range(k):
                clause = []
                # 编码这个颜色
                for b in range(bits):
                    bit_value = (color >> b) & 1
                    # 如果u有这个颜色的第b位，那么v不能有相同的位
                    literal_u = var_id(u, b) if bit_value == 1 else -var_id(u, b)
                    literal_v = var_id(v, b) if bit_value == 1 else -var_id(v, b)
                    # 添加约束：不能u和v同时有这个颜色的编码
                    clauses.append([-literal_u, -literal_v])
        
        n_vars = n_nodes * bits
        n_clauses = len(clauses)
        
        dimacs = f"p cnf {n_vars} {n_clauses}\n"
        for clause in clauses:
            dimacs += " ".join(str(lit) for lit in clause) + " 0\n"
        
        return dimacs
    
    def solve_cnf(self, cnf_str: str, timeout: Optional[int] = None) -> Tuple[Optional[bool], Optional[List[int]], Dict[str, Any]]:
        """
        求解SAT问题
        
        Args:
            cnf_str: DIMACS格式的CNF字符串
            timeout: 超时时间（秒），覆盖配置中的设置
            
        Returns:
            (is_satisfiable, assignment, stats) 
            如果可满足则返回赋值，否则为None
            stats包含求解统计信息
        """
        start_time = time.time()
        timeout = timeout or self.config.timeout
        
        # 创建临时文件
        temp_dir = tempfile.mkdtemp(dir=self.config.temp_dir)
        cnf_file = os.path.join(temp_dir, "problem.cnf")
        out_file = os.path.join(temp_dir, "output.txt")
        
        try:
            # 写入CNF文件
            with open(cnf_file, 'w') as f:
                f.write(cnf_str)
            
            # 构建命令
            cmd = [self.solver_path, cnf_file]
            if self.config.verbose:
                cmd.append("--verbose")
            
            # 运行SAT求解器
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                output = result.stdout
                stderr = result.stderr
                
                # 记录统计信息
                self.stats["total_solves"] += 1
                solve_time = time.time() - start_time
                self.stats["total_time"] += solve_time
                
                stats = {
                    "time": solve_time,
                    "return_code": result.returncode,
                    "stdout": output,
                    "stderr": stderr
                }
                
                # 解析输出
                if "s SATISFIABLE" in output:
                    self.stats["satisfiable"] += 1
                    assignment = self._parse_assignment(output, len(cnf_str))
                    return True, assignment, stats
                elif "s UNSATISFIABLE" in output:
                    self.stats["unsatisfiable"] += 1
                    return False, None, stats
                else:
                    # 无法解析输出，可能是求解器错误
                    warnings.warn(f"Could not parse solver output: {output[:200]}")
                    return None, None, stats
                    
            except subprocess.TimeoutExpired:
                self.stats["timeouts"] += 1
                stats = {
                    "time": timeout,
                    "timeout": True,
                    "return_code": None
                }
                return None, None, stats
                
        finally:
            # 清理临时文件
            if not self.config.keep_temp_files:
                shutil.rmtree(temp_dir, ignore_errors=True)
            elif self.config.verbose:
                print(f"Temporary files kept in: {temp_dir}")
    
    def _parse_assignment(self, output: str, n_vars_hint: int = 0) -> List[int]:
        """从求解器输出解析赋值"""
        assignment = []
        
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('v'):
                # 解析赋值行，如 "v 1 -2 3 0"
                parts = line.split()
                for part in parts[1:]:  # 跳过'v'
                    if part == '0':
                        break
                    try:
                        lit = int(part)
                        assignment.append(lit)
                    except ValueError:
                        continue
        
        # 如果解析失败，尝试其他格式
        if not assignment and n_vars_hint > 0:
            # 有些求解器可能使用不同格式
            lines = output.split('\n')
            for line in lines:
                if line and not line.startswith(('c', 's', 'v')):
                    parts = line.split()
                    for part in parts:
                        if part == '0':
                            break
                        try:
                            lit = int(part)
                            if abs(lit) <= n_vars_hint:
                                assignment.append(lit)
                        except ValueError:
                            continue
        
        return assignment
    
    def is_k_colorable(self, graph: UnitDistanceGraph, k: int, 
                       timeout: Optional[int] = None,
                       encoding: str = "standard") -> Tuple[Optional[bool], Optional[Dict[int, int]], Dict[str, Any]]:
        """
        判断图是否k可染色
        
        Args:
            graph: 单位距离图
            k: 颜色数
            timeout: 超时时间
            encoding: 编码方式
            
        Returns:
            (is_colorable, coloring, stats) 
            如果可染色则返回染色方案，否则为None
        """
        if k <= 0:
            raise ValueError(f"Number of colors must be positive, got {k}")
        
        if graph.num_nodes == 0:
            # 空图总是可染色的
            return True, {}, {"empty_graph": True, "time": 0.0}
        
        # 转换为CNF
        cnf = self.graph_to_cnf(graph, k, encoding)
        
        # 求解
        satisfiable, assignment, stats = self.solve_cnf(cnf, timeout)
        
        if satisfiable is None:  # 超时或错误
            return None, None, stats
        elif satisfiable:
            # 解析染色方案
            coloring = self._assignment_to_coloring(assignment, graph.num_nodes, k, encoding)
            return True, coloring, stats
        else:
            return False, None, stats
    
    def _assignment_to_coloring(self, assignment: List[int], n_nodes: int, 
                               k: int, encoding: str = "standard") -> Dict[int, int]:
        """将SAT赋值转换为染色方案"""
        coloring = {}
        
        if encoding == "log":
            # 对数编码
            bits = max(1, int(np.ceil(np.log2(k))))
            for i in range(n_nodes):
                color_code = 0
                for b in range(bits):
                    var_id = i * bits + b + 1
                    if var_id in assignment:
                        color_code |= (1 << b)
                    elif -var_id in assignment:
                        pass  # 位为0
                    else:
                        # 变量未赋值，使用默认值
                        pass
                if color_code < k:
                    coloring[i] = color_code
                else:
                    # 无效颜色，使用默认
                    coloring[i] = 0
        else:
            # 标准或高效编码
            for i in range(n_nodes):
                for c in range(k):
                    var_id = i * k + c + 1
                    if var_id in assignment:
                        coloring[i] = c
                        break
                # 如果没有找到颜色，使用默认
                if i not in coloring:
                    coloring[i] = 0
        
        return coloring
    
    def estimate_chromatic_number(self, graph: UnitDistanceGraph, 
                                 max_k: int = 10,
                                 timeout_per_test: int = 30) -> Tuple[int, Dict[str, Any]]:
        """
        估计图的色数
        
        Args:
            graph: 单位距离图
            max_k: 最大测试的颜色数
            timeout_per_test: 每个测试的超时时间
            
        Returns:
            (chromatic_number, stats) 
            估计的色数（如果所有测试都超时则返回-1）
        """
        stats = {
            "tests": [],
            "lower_bound": 1,
            "upper_bound": max_k,
            "determined": False
        }
        
        # 首先测试下界
        for k in range(1, max_k + 1):
            print(f"测试 {k}-染色...")
            start_time = time.time()
            
            colorable, coloring, solve_stats = self.is_k_colorable(
                graph, k, timeout=timeout_per_test
            )
            
            test_time = time.time() - start_time
            
            test_result = {
                "k": k,
                "colorable": colorable,
                "time": test_time,
                "stats": solve_stats
            }
            stats["tests"].append(test_result)
            
            if colorable is None:  # 超时
                print(f"  {k}-染色测试超时 ({test_time:.1f}秒)")
                stats["upper_bound"] = min(stats["upper_bound"], k)
                # 继续测试下一个k，但标记为不确定
                continue
            elif not colorable:
                print(f"  {k}-染色: 否 ({test_time:.1f}秒)")
                stats["lower_bound"] = max(stats["lower_bound"], k + 1)
                if k == max_k:
                    return k + 1, stats
            else:
                print(f"  {k}-染色: 是 ({test_time:.1f}秒)")
                stats["upper_bound"] = min(stats["upper_bound"], k)
                if k == 1:
                    return 1, stats
        
        # 确定色数范围
        if stats["lower_bound"] <= stats["upper_bound"]:
            if stats["lower_bound"] == stats["upper_bound"]:
                stats["determined"] = True
                return stats["lower_bound"], stats
            else:
                # 返回下界（保守估计）
                return stats["lower_bound"], stats
        else:
            # 矛盾，返回-1表示不确定
            return -1, stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取求解器统计信息"""
        stats = self.stats.copy()
        if stats["total_solves"] > 0:
            stats["avg_time"] = stats["total_time"] / stats["total_solves"]
            stats["satisfiable_rate"] = stats["satisfiable"] / stats["total_solves"]
            stats["unsatisfiable_rate"] = stats["unsatisfiable"] / stats["total_solves"]
            stats["timeout_rate"] = stats["timeouts"] / stats["total_solves"]
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            "total_solves": 0,
            "satisfiable": 0,
            "unsatisfiable": 0,
            "timeouts": 0,
            "total_time": 0.0
        }

# 预配置的求解器实例
def create_solver(solver_name: str = "kissat", timeout: int = 60) -> SATSolver:
    """创建预配置的SAT求解器"""
    config = SATConfig(solver_name=solver_name, timeout=timeout)
    return SATSolver(config)

KISSAT_SOLVER = None
GLUCOSE_SOLVER = None
MINISAT_SOLVER = None

try:
    KISSAT_SOLVER = create_solver("kissat", timeout=60)
except RuntimeError:
    pass 

try:
    GLUCOSE_SOLVER = create_solver("glucose", timeout=60)
except RuntimeError:
    pass  

try:
    MINISAT_SOLVER = create_solver("minisat", timeout=60)
except RuntimeError:
    pass  