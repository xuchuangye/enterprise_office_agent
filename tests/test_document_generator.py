"""
测试文档生成
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
from tools.document_generator import DocumentGeneratorTool

# 创建大模型实例
llm = Tongyi(api_key=api_key, model_name="qwen-plus")

# 创建Agent实例
agent = OfficeAgent(llm=llm, enable_planner=True)

# 创建工具实例
document_generator = DocumentGeneratorTool()

# 注册工具
agent.register_tool(document_generator)

# 生成会议纪要
meeting_notes= document_generator.invoke(
    {
        "doc_type":"meeting_notes",
        "title":"会议纪要",
        "participants":["张三", "李四", "王五"],
        "agenda":["项目进度", "项目风险", "项目计划"],
        "content":"会议内容",
        "auto_generate":True, # True表示开启大模型自动生成
    }
)

# 测试生成会议纪要
print(f"生成会议纪要结果：\n{meeting_notes}")


