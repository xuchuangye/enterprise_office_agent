"""
大语言模型的接入
"""
import os
from dashscope import Generation
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()

# 获取API Key
api_key = os.getenv("DASHSCOPE_API_KEY")

# 测试不同的temperature值
temperatures = [0.0, 0.3, 0.7, 1.0]
prompt = "给我一个关于人工智能的创意标题"

for temperature in temperatures:
    print(f"{'='*20} {temperature} {'='*20}")
    response = Generation.call(
        model='qwen-plus',
        api_key=api_key,
        prompt=prompt,
        temperature=temperature,
    )

    # 安全访问响应（不依赖任何类型定义）
    if hasattr(response, "status_code") and response.status_code == 200:
        print(response.output.text)  # type: ignore # 直接使用属性
    else:
        print(f"请求失败: {getattr(response, 'code', 'N/A')} - {getattr(response, 'message', '未知错误')}")


    if response.status_code == 200: 
        print(response.output.text) 