"""
LangChain核心组件
"""
import os
from dotenv import load_dotenv
from langchain_community.llms import Tongyi

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

# 创建大模型实例
llm = Tongyi(
    model="qwen-plus",
    api_key=api_key,
    temperature=0.7,
    max_tokens=2048,
    streaming=True
)

# 测试调用
response = llm.invoke("解释一下什么是LangChain")
print(response)