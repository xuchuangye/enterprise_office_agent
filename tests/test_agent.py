"""
Agent
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
from langchain_core.tools import Tool
from langchain_core.tools import StructuredTool

# 办公助手Agent
from agent_core.agent import OfficeAgent
from tools.knowledge_qa_tool import KnowledgeQATool

from pydantic import BaseModel, Field

# 导入日志
from config.logging_config import setup_logging
# 初始化日志
setup_logging()

# 创建大语言模型实例
llm = Tongyi(model_name="qwen-plus", api_key=api_key)

# 创建Agent实例
# 开启强制使用任务规划器
agent = OfficeAgent(llm=llm, enable_planner=True)

# 生成工具方法
def hello_tool(name: str) -> str:
    return f"你好，{name}。我是白泽，很高兴认识你。"

# 创建工具实例
tool = Tool(
    name="hello", 
    func=hello_tool, 
    description="向指定的人物打招呼，输入参数是人物的名字。"
)


# 注册工具
agent.register_tool(tool)

# 开启日志
import logging
logger = logging.getLogger(__name__)

# 使用Agent
# 测试简单任务
print("=========================测试简单任务=========================")
logger.info("=========================测试简单任务=========================")
result = agent.run("请向张三打个招呼。")
print("问答：", result["output"])
logger.info(f"问答：{result['output']}")
print("中间步骤：", result["intermediate_steps"])
logger.info(f"中间步骤：{result['intermediate_steps']}")

# 测试复杂任务
print("=========================测试复杂任务=========================")
logger.info("=========================测试复杂任务=========================")
result = agent.run("请帮我分析一下这个任务。")
print("问答：", result["output"])
logger.info(f"问答：{result['output']}")
print("中间步骤：", result["intermediate_steps"])
logger.info(f"中间步骤：{result['intermediate_steps']}")

# 测试记忆功能
print("=========================测试记忆功能=========================")
logger.info("=========================测试记忆功能=========================")
session_id = "test_001"
result = agent.run("记住我是李四", session_id=session_id)
print("问答：", result["output"])
logger.info(f"问答：{result['output']}")

result = agent.run("我是谁。", session_id=session_id)
print("问答：", result["output"])
logger.info(f"问答：{result['output']}")

# 清理会话
agent.clear_session(session_id)

# 天气参数输入的类
class WeatherInput(BaseModel):
    city: str = Field(..., description="城市名称，例如：北京、上海、广州等。")
    date: str = Field(default="今天", description="日期，例如：2024-06-01。")

# 定义查询天气的工具
def get_weather_tool(city: str, date: str = "今天") -> str:
    # 这里可以调用实际的天气API获取天气信息
    return f"{city}{date}的天气是晴天，温度25摄氏度。"

get_weather_tool = StructuredTool.from_function(
    func = get_weather_tool,
    name = "get_weather",
    description = "获取指定城市指定日期的天气信息。",
    args_schema =WeatherInput,
    )
weather_tool = [get_weather_tool]
agent.register_tools(weather_tool)

# 测试简单任务
print("=========================测试简单任务=========================")
result = agent.run("查询北京的天气")
print("问答：", result["output"])
print("中间步骤：", result["intermediate_steps"])