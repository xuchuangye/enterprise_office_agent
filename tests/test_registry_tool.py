"""
测试工具的注册
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
from tools.calculator import CalculatorTool

# 创建大模型实例
llm = Tongyi(api_key=api_key, model_name="qwen-plus", temperature=0.3)

# 创建Agent实例，强制使用规划器
agent = OfficeAgent(llm=llm, enable_planner=True)

# 注册工具
agent.register_tools([
    CalculatorTool(),
])

# 已注册的工具
for tool in agent.tool_registry.list_tools():
    print(f"已注册工具：{tool.name} - {tool.description}")


# 测试
result = agent.run("请计算((1+2+3)*(2+5))的2次方，并给出计算过程")
print(result["output"])
