"""
交互式绘图 - 完整实现（需要ipywidgets）
"""
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    HAS_INTERACTIVE = True
except ImportError:
    HAS_INTERACTIVE = False
    print("Note: Interactive features require ipywidgets. Install with: pip install ipywidgets")

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

from ..core.graph_builder import UnitDistanceGraph, GraphBuilder
from ..core.geometry_engine import GeometryEngine
from ..utils.data_io import load_graph, save_graph

class InteractivePlotter:
    """交互式绘图器"""
    
    def __init__(self, results_dir: str = "results"):
        """
        初始化交互式绘图器
        
        Args:
            results_dir: 结果目录
        """
        if not HAS_INTERACTIVE:
            raise ImportError("Interactive features require ipywidgets")
        
        self.results_dir = Path(results_dir)
        self.current_graph = None
        self.current_geometry = None
        
        # 创建输出区域
        self.output = widgets.Output()
    
    def create_experiment_selector(self, 
                                  on_select: Optional[Callable] = None) -> widgets.Widget:
        """
        创建实验选择器
        
        Args:
            on_select: 选择实验时的回调函数
            
        Returns:
            ipywidgets组件
        """
        # 列出实验
        exp_dirs = list(self.results_dir.glob("*"))
        experiment_names = [d.name for d in exp_dirs if d.is_dir()]
        
        if not experiment_names:
            experiment_names = ["No experiments found"]
        
        # 创建下拉菜单
        dropdown = widgets.Dropdown(
            options=experiment_names,
            description='Experiment:',
            disabled=False,
            layout=widgets.Layout(width='80%')
        )
        
        # 创建加载按钮
        load_button = widgets.Button(
            description='Load Experiment',
            button_style='primary',
            layout=widgets.Layout(width='20%')
        )
        
        # 创建水平布局
        selector = widgets.HBox([dropdown, load_button])
        
        def on_load_button_clicked(b):
            experiment_name = dropdown.value
            if experiment_name != "No experiments found":
                with self.output:
                    clear_output(wait=True)
                    print(f"Loading experiment: {experiment_name}")
                    
                    try:
                        # 加载实验
                        from ..pipeline.result_handler import ResultHandler
                        handler = ResultHandler(str(self.results_dir))
                        result = handler.load_result(experiment_name, load_graph=True)
                        
                        if "graph" in result:
                            self.current_graph = result["graph"]
                            
                            # 创建几何引擎
                            config = result.get("config", {})
                            geometry_config = config.get("geometry", {})
                            self.current_geometry = GeometryEngine(
                                lattice_type=geometry_config.get("lattice_type", "hexagonal")
                            )
                            
                            # 显示图
                            from .graph_visualizer import GraphVisualizer
                            visualizer = GraphVisualizer()
                            
                            coloring = result.get("coloring")
                            fig = visualizer.plot_graph(
                                self.current_graph, 
                                self.current_geometry,
                                coloring=coloring,
                                title=f"Experiment: {experiment_name}"
                            )
                            plt.show()
                            
                            # 显示实验信息
                            print(f"Experiment: {experiment_name}")
                            print(f"Success: {result.get('success', False)}")
                            print(f"Chromatic number: {result.get('chromatic_number', -1)}")
                            print(f"Graph: {self.current_graph.num_nodes} nodes, "
                                  f"{self.current_graph.num_edges} edges")
                            print(f"Epsilon: {self.current_graph.epsilon}")
                            
                            # 调用回调函数
                            if on_select:
                                on_select(result)
                        else:
                            print("No graph found in experiment")
                            
                    except Exception as e:
                        print(f"Error loading experiment: {e}")
        
        load_button.on_click(on_load_button_clicked)
        
        return widgets.VBox([selector, self.output])
    
    def create_graph_explorer(self, graph: UnitDistanceGraph, 
                            geometry: GeometryEngine) -> widgets.Widget:
        """
        创建图浏览器
        
        Args:
            graph: 单位距离图
            geometry: 几何引擎
            
        Returns:
            ipywidgets组件
        """
        self.current_graph = graph
        self.current_geometry = geometry
        
        # 创建控件
        show_labels = widgets.Checkbox(
            value=False,
            description='Show node labels',
            disabled=False
        )
        
        node_size = widgets.IntSlider(
            value=50,
            min=10,
            max=200,
            step=10,
            description='Node size:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        
        edge_alpha = widgets.FloatSlider(
            value=0.5,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Edge alpha:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f'
        )
        
        show_domain = widgets.Checkbox(
            value=True,
            description='Show fundamental domain',
            disabled=False
        )
        
        plot_button = widgets.Button(
            description='Update Plot',
            button_style='primary'
        )
        
        # 创建输出区域
        plot_output = widgets.Output()
        
        def on_plot_button_clicked(b):
            with plot_output:
                clear_output(wait=True)
                
                from .graph_visualizer import GraphVisualizer
                visualizer = GraphVisualizer()
                
                fig = visualizer.plot_graph(
                    self.current_graph,
                    self.current_geometry,
                    title="Interactive Graph Explorer",
                    show_labels=show_labels.value,
                    node_size=node_size.value,
                    edge_alpha=edge_alpha.value,
                    show_domain=show_domain.value
                )
                
                plt.show()
        
        plot_button.on_click(on_plot_button_clicked)
        
        # 创建布局
        controls = widgets.VBox([
            widgets.HTML("<h3>Graph Explorer</h3>"),
            show_labels,
            node_size,
            edge_alpha,
            show_domain,
            plot_button
        ])
        
        return widgets.VBox([controls, plot_output])
    
    def create_degree_analyzer(self, graph: UnitDistanceGraph) -> widgets.Widget:
        """
        创建度分析器
        
        Args:
            graph: 单位距离图
            
        Returns:
            ipywidgets组件
        """
        self.current_graph = graph
        
        # 创建控件
        num_bins = widgets.IntSlider(
            value=10,
            min=5,
            max=50,
            step=5,
            description='Number of bins:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        
        plot_button = widgets.Button(
            description='Plot Degree Distribution',
            button_style='primary'
        )
        
        # 创建输出区域
        plot_output = widgets.Output()
        
        def on_plot_button_clicked(b):
            with plot_output:
                clear_output(wait=True)
                
                from .graph_visualizer import GraphVisualizer
                visualizer = GraphVisualizer()
                
                fig = visualizer.plot_degree_distribution(
                    self.current_graph,
                    figsize=(12, 5)
                )
                
                plt.show()
        
        plot_button.on_click(on_plot_button_clicked)
        
        # 创建布局
        controls = widgets.VBox([
            widgets.HTML("<h3>Degree Analysis</h3>"),
            num_bins,
            plot_button
        ])
        
        return widgets.VBox([controls, plot_output])
    
    def create_cover_explorer(self, graph: UnitDistanceGraph,
                            geometry: GeometryEngine) -> widgets.Widget:
        """
        创建覆盖空间浏览器
        
        Args:
            graph: 单位距离图
            geometry: 几何引擎
            
        Returns:
            ipywidgets组件
        """
        self.current_graph = graph
        self.current_geometry = geometry
        
        # 创建控件
        u_copies = widgets.IntSlider(
            value=3,
            min=1,
            max=7,
            step=1,
            description='U copies:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        
        v_copies = widgets.IntSlider(
            value=3,
            min=1,
            max=7,
            step=1,
            description='V copies:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        
        node_size = widgets.IntSlider(
            value=30,
            min=10,
            max=100,
            step=10,
            description='Node size:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        
        edge_alpha = widgets.FloatSlider(
            value=0.3,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Edge alpha:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f'
        )
        
        plot_button = widgets.Button(
            description='Plot Cover Space',
            button_style='primary'
        )
        
        # 创建输出区域
        plot_output = widgets.Output()
        
        def on_plot_button_clicked(b):
            with plot_output:
                clear_output(wait=True)
                
                from .graph_visualizer import GraphVisualizer
                visualizer = GraphVisualizer()
                
                fig = visualizer.plot_cover(
                    self.current_graph,
                    self.current_geometry,
                    num_copies=(u_copies.value, v_copies.value),
                    node_size=node_size.value,
                    edge_alpha=edge_alpha.value
                )
                
                plt.show()
        
        plot_button.on_click(on_plot_button_clicked)
        
        # 创建布局
        controls = widgets.VBox([
            widgets.HTML("<h3>Cover Space Explorer</h3>"),
            u_copies,
            v_copies,
            node_size,
            edge_alpha,
            plot_button
        ])
        
        return widgets.VBox([controls, plot_output])
    
    def create_dashboard(self, graph: UnitDistanceGraph, 
                        geometry: GeometryEngine) -> widgets.Widget:
        """
        创建完整仪表板
        
        Args:
            graph: 单位距离图
            geometry: 几何引擎
            
        Returns:
            ipywidgets选项卡组件
        """
        self.current_graph = graph
        self.current_geometry = geometry
        
        # 创建各个选项卡的内容
        graph_explorer = self.create_graph_explorer(graph, geometry)
        degree_analyzer = self.create_degree_analyzer(graph)
        cover_explorer = self.create_cover_explorer(graph, geometry)
        
        # 创建选项卡
        tab = widgets.Tab()
        tab.children = [graph_explorer, degree_analyzer, cover_explorer]
        tab.set_title(0, 'Graph Explorer')
        tab.set_title(1, 'Degree Analysis')
        tab.set_title(2, 'Cover Space')
        
        return tab

def create_interactive_demo():
    """创建交互式演示"""
    if not HAS_INTERACTIVE:
        print("Interactive features require ipywidgets. Install with: pip install ipywidgets")
        return None
    
    # 创建一个简单的示例图
    geometry = GeometryEngine()
    builder = GraphBuilder(geometry)
    
    # 生成一些点
    nodes = builder.initialize_points_fibonacci(20)
    graph = builder.construct_graph(nodes)
    
    # 创建交互式绘图器
    plotter = InteractivePlotter()
    
    # 创建仪表板
    dashboard = plotter.create_dashboard(graph, geometry)
    
    return dashboard