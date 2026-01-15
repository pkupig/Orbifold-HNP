"""
pytest配置和共享fixture - 完整实现
"""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# 添加src目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.geometry_engine import GeometryEngine, LatticeConfig
from src.core.graph_builder import GraphBuilder, GraphConfig, UnitDistanceGraph
from src.core.sat_solver import SATSolver, SATConfig
from src.core.optimizer import GraphOptimizer, OptimizerConfig

@pytest.fixture
def geometry_engine():
    """几何引擎fixture"""
    return GeometryEngine(LatticeConfig(type="hexagonal"))

@pytest.fixture
def square_geometry_engine():
    """正方形晶格几何引擎fixture"""
    return GeometryEngine(LatticeConfig(type="square"))

@pytest.fixture
def graph_builder(geometry_engine):
    """图构建器fixture"""
    config = GraphConfig(epsilon=0.05, use_kdtree=False)
    return GraphBuilder(geometry_engine, config)

@pytest.fixture
def simple_graph(graph_builder):
    """简单图fixture"""
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    return graph_builder.construct_graph(nodes, use_kdtree=False)

@pytest.fixture
def sat_solver():
    """SAT求解器fixture"""
    # 检查是否有可用的求解器
    solver_names = ["kissat", "glucose", "minisat"]
    available_solver = None
    
    for solver_name in solver_names:
        try:
            solver = SATSolver(SATConfig(solver_name=solver_name, timeout=2))
            # 测试求解器
            test_cnf = "p cnf 1 1\n1 0\n"
            result, _, _ = solver.solve_cnf(test_cnf)
            if result is not None:
                available_solver = solver_name
                break
        except:
            continue
    
    if available_solver is None:
        pytest.skip("No SAT solver available")
    
    return SATSolver(SATConfig(solver_name=available_solver, timeout=5))

@pytest.fixture
def optimizer(geometry_engine, graph_builder):
    """优化器fixture"""
    config = OptimizerConfig(
        method="constraint_based",
        num_candidates=10,
        relaxation_iterations=2
    )
    return GraphOptimizer(geometry_engine, graph_builder, config)

@pytest.fixture
def test_coloring():
    """测试染色方案fixture"""
    return {0: 0, 1: 1, 2: 0, 3: 1}

@pytest.fixture
def temp_dir():
    """临时目录fixture"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def pipeline_config():
    """管道配置fixture"""
    return {
        "geometry": {
            "lattice_type": "hexagonal",
            "v1": [1.0, 0.0],
            "v2": [0.5, 0.86602540378]
        },
        "graph_builder": {
            "epsilon": 0.05,
            "sampling_method": "fibonacci",
            "use_kdtree": False
        },
        "sat_solver": {
            "solver_name": "kissat",
            "timeout": 5,
            "verbose": False
        },
        "optimizer": {
            "method": "constraint_based",
            "num_candidates": 10,
            "relaxation_iterations": 2
        },
        "pipeline": {
            "initial_points": 5,
            "target_k": 3,
            "max_iterations": 3,
            "anneal_epsilon": False
        }
    }

# 标记某些测试为慢速测试
def pytest_addoption(parser):
    """添加pytest命令行选项"""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="运行慢速测试"
    )

def pytest_configure(config):
    """配置pytest"""
    config.addinivalue_line(
        "markers", "slow: 标记测试为慢速测试"
    )

def pytest_collection_modifyitems(config, items):
    """根据命令行选项跳过慢速测试"""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="需要 --run-slow 选项来运行慢速测试")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)