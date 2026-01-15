"""
管道测试 - 完整实现
"""
import unittest
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.coloring_pipeline import OrbifoldColoringPipeline, PipelineResult
from src.pipeline.experiment_runner import ExperimentRunner, ExperimentConfig
from src.core.graph_builder import GraphBuilder, UnitDistanceGraph

class TestOrbifoldColoringPipeline(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        # 使用内存中的配置，避免文件依赖
        self.config = {
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
                "target_k": 3,  # 较小的k，便于测试
                "max_iterations": 3,
                "anneal_epsilon": False
            }
        }
        
        self.pipeline = OrbifoldColoringPipeline(config_dict=self.config)
    
    def test_pipeline_creation(self):
        """测试管道创建"""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.geometry)
        self.assertIsNotNone(self.pipeline.builder)
        self.assertIsNotNone(self.pipeline.solver)
        self.assertIsNotNone(self.pipeline.optimizer)
        
        # 检查配置
        self.assertEqual(
            self.pipeline.config["geometry"]["lattice_type"],
            "hexagonal"
        )
        self.assertEqual(
            self.pipeline.config["pipeline"]["target_k"],
            3
        )
    
    def test_initialize_graph(self):
        """测试图初始化"""
        # 初始时图应该为None
        self.assertIsNone(self.pipeline.graph)
        
        # 初始化图
        self.pipeline.initialize_graph()
        
        # 图应该被创建
        self.assertIsNotNone(self.pipeline.graph)
        self.assertGreater(self.pipeline.graph.num_nodes, 0)
        
        # 检查历史
        self.assertGreater(len(self.pipeline.history), 0)
        self.assertEqual(self.pipeline.history[0]["action"], "initialize")
    
    def test_update_config(self):
        """测试配置更新"""
        # 更新配置
        updates = {
            "pipeline": {
                "initial_points": 10,
                "target_k": 4
            }
        }
        
        self.pipeline.update_config(updates)
        
        # 检查更新
        self.assertEqual(
            self.pipeline.config["pipeline"]["initial_points"],
            10
        )
        self.assertEqual(
            self.pipeline.config["pipeline"]["target_k"],
            4
        )
    
    def test_run_iteration(self):
        """测试运行迭代"""
        # 首先初始化图
        self.pipeline.initialize_graph()
        
        initial_nodes = self.pipeline.graph.num_nodes
        
        # 运行一次迭代
        status, coloring, message = self.pipeline.run_iteration()
        
        # 检查结果
        self.assertIsNotNone(status)
        self.assertIsNotNone(message)
        
        # 如果可染色，应该添加了新点
        if status:
            self.assertGreater(self.pipeline.graph.num_nodes, initial_nodes)
        
        # 检查历史更新
        self.assertGreater(len(self.pipeline.history), 1)
        self.assertEqual(self.pipeline.history[-1]["action"], "add_point")
    
    def test_estimate_chromatic_number(self):
        """测试色数估计"""
        # 初始化图
        self.pipeline.initialize_graph()
        
        # 估计色数
        chromatic_num, stats = self.pipeline.estimate_chromatic_number(max_k=4)
        
        # 检查结果
        self.assertIsNotNone(chromatic_num)
        self.assertIsNotNone(stats)
        
        # 色数应该在合理范围内
        self.assertGreaterEqual(chromatic_num, 1)
        self.assertLessEqual(chromatic_num, 4)
    
    def test_save_load_state(self):
        """测试状态保存和加载"""
        # 初始化并运行一些迭代
        self.pipeline.initialize_graph()
        self.pipeline.run_iteration()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            state_file = f.name
            
            try:
                # 保存状态
                self.pipeline.save_state(state_file)
                
                # 创建新管道并加载状态
                new_pipeline = OrbifoldColoringPipeline(config_dict=self.config)
                new_pipeline.load_state(state_file)
                
                # 检查状态恢复
                self.assertEqual(
                    new_pipeline.graph.num_nodes,
                    self.pipeline.graph.num_nodes
                )
                self.assertEqual(
                    len(new_pipeline.history),
                    len(self.pipeline.history)
                )
                self.assertEqual(
                    new_pipeline.current_iteration,
                    self.pipeline.current_iteration
                )
                
            finally:
                os.unlink(state_file)
    
    def test_pipeline_result(self):
        """测试管道结果"""
        result = PipelineResult(
            success=True,
            chromatic_number=4,
            final_graph_size=(10, 15),
            iterations=5,
            history=[{"test": "data"}],
            config=self.config,
            runtime=12.5,
            termination_reason="counterexample_found"
        )
        
        # 测试转换为字典
        result_dict = result.to_dict()
        
        self.assertTrue(result_dict["success"])
        self.assertEqual(result_dict["chromatic_number"], 4)
        self.assertEqual(result_dict["final_graph_size"], [10, 15])
        self.assertEqual(result_dict["iterations"], 5)
        self.assertEqual(result_dict["runtime"], 12.5)
        
        # 测试保存
        with tempfile.TemporaryDirectory() as temp_dir:
            result_file = os.path.join(temp_dir, "result.json")
            
            # 需要有一个图才能保存
            # 创建一个简单图
            geometry = self.pipeline.geometry
            builder = GraphBuilder(geometry)
            nodes = builder.initialize_points(5)
            graph = builder.construct_graph(nodes)
            
            result.graph = graph
            result.save(result_file)
            
            # 检查文件存在
            self.assertTrue(os.path.exists(result_file))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "graph.pkl")))

class TestExperimentRunner(unittest.TestCase):
    
    def test_experiment_config(self):
        """测试实验配置"""
        config = ExperimentConfig(
            name="test_experiment",
            config={"test": "value"},
            description="Test experiment",
            enabled=True,
            priority=1
        )
        
        self.assertEqual(config.name, "test_experiment")
        self.assertEqual(config.config, {"test": "value"})
        self.assertEqual(config.description, "Test experiment")
        self.assertTrue(config.enabled)
        self.assertEqual(config.priority, 1)
    
    def test_experiment_result(self):
        """测试实验结果"""
        from src.pipeline.coloring_pipeline import PipelineResult
        
        pipeline_result = PipelineResult(
            success=True,
            chromatic_number=4,
            final_graph_size=(10, 15),
            iterations=5,
            history=[],
            config={},
            runtime=10.0
        )
        
        result = ExperimentResult(
            experiment_name="test",
            pipeline_result=pipeline_result,
            success=True
        )
        
        self.assertEqual(result.experiment_name, "test")
        self.assertEqual(result.pipeline_result, pipeline_result)
        self.assertTrue(result.success)
        self.assertGreater(result.runtime, 0)

if __name__ == '__main__':
    unittest.main()