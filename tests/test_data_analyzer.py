"""
测试读取数据分析工具
"""

import os
import sys
from pathlib import Path
# 1. 获取当前文件(test_agent.py)的绝对路径
current_file = Path(__file__).resolve()
# 2. 向上回退两级目录: tests -> enterprise_office_agent (项目根目录)
project_root = current_file.parent.parent
# 3. 将项目根目录插入到系统路径的最前面(索引0)
# 这确保了Python会优先在你的项目中查找模块，而不是去site-packages里找
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
# 加载环境变量
load_dotenv()
# 获取API Key
api_key = os.getenv("DASHSCOPE_API_KEY")

from langchain_community.llms import Tongyi

from agent_core.agent import OfficeAgent
from tools.data_analyzer import DataAnalyzerTool

# 创建大模型实例
llm = Tongyi(api_key=api_key, model_name="qwen-plus")

# 创建Agent实例
agent = OfficeAgent(llm=llm, enable_planner=True)

# 创建工具实例
analyzer = DataAnalyzerTool()

# 注册工具
# agent.register_tools([analyzer])

# 测试用例
# 测试读取数据
read_data = analyzer.invoke(
    {
        "action": "read_data",
        "file_path": "data/sales_data.xlsx",
    }
)
print(f"\n读取数据结果: {read_data}")


# 测试清洗数据
clean_data = analyzer.invoke(
    {
        "action": "clean_data", # 内部使用变量_current_file作为中间状态，存储当前读取数据的文件路径
        # "file_path": "data/sales_data.xlsx",
    }
)
print(f"\n清洗数据结果: {clean_data}")

# 测试分析数据
analyzer_data = analyzer.invoke(
    {
        "action": "analyze", # 内部使用变量_current_file作为中间状态，存储当前读取数据的文件路径
        # "file_path": "data/sales_data.xlsx", 
    }
)
print(f"\n分析数据结果: {analyzer_data}")

# 测试可视化数据
visualization_data = analyzer.invoke(
    {
        "action": "visualize", # 内部使用变量_current_file作为中间状态，存储当前读取数据的文件路径
        # "file_path": "data/sales_data.xlsx",
    }
)
print(f"\n可视化数据结果: {visualization_data}")

# 测试生成报告
report = analyzer.invoke(
    {
        "action": "generate_report", # 内部使用变量_current_file作为中间状态，存储当前读取数据的文件路径
        # "file_path": "data/sales_data.xlsx",
        "auto_generate_doc": True, # 是否自动生成Word文档报告
    }
)
print(f"\n生成报告结果: {report}")