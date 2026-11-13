"""
测试上下文记忆
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

from langchain_community.llms import Tongyi
from langchain_core.tools import Tool
from agent_core.agent import OfficeAgent
from tools.data_analyzer import DataAnalyzerTool
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()
# 获取API Key
api_key = os.getenv("DASHSCOPE_API_KEY")

# 创建大模型实例
llm = Tongyi(model_name="qwen-plus", api_key=api_key)

# 创建Agent实例
agent = OfficeAgent(llm=llm, enable_planner=True)

# 创建工具实例
data_analyzer = DataAnalyzerTool()

# 注册工具
agent.register_tools([data_analyzer])

# 测试记忆功能
session_id = "test_session001"

print(f"测试第一轮对话：\n")
result = agent.run("你好，我叫小明。", session_id=session_id)
print(f"{result['output']}\n")

print(f"测试第二轮对话：\n")
result = agent.run("我想学习AI", session_id=session_id)
print(f"{result['output']}\n")

print(f"测试第三轮对话：\n")
result = agent.run("我叫什么？刚才说了想学习什么？", session_id=session_id)
print(f"{result['output']}\n")

# 查看记忆会话历史
history = agent.memory_manager.get_messages(session_id)
for i, message in enumerate(history, 1):
    print(f"[{i}]{message['role']}: {message['content'][:50]}...")