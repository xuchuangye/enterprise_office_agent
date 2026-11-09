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

from agent_core.agent import TaskPlanner

# 创建大语言模型实例
llm = Tongyi(
    api_key=api_key, 
    model_name="qwen-plus",
    temperature=0.3
)

# 创建Planner实例
# llm=llm就是主动策略：大语言模型动态规划模式
# llm=None就是备用的降级策略，预设规则规划模式
# planner = TaskPlanner(llm=llm)
# planner = TaskPlanner(llm=llm)
planner = TaskPlanner(llm=None)

# 测试用例
test_cases = [
    "帮我分析data/sales_data.xlsx文件，生成销售分析报告",
    "生成一份会议纪要",
    "帮我处理一个报销审核",
    "查询公司的差旅政策",
]

for case in test_cases:
    print(f"测试用例：{case}")
    plan = planner.plan(case, ["document_generator", "api_tool", "data_analyzer", "knowledge_qa"])
    print(f"生成 {len(plan)} 个步骤:")
    for step in plan:
        print(f"- {step['tool']}: {step['params']}")