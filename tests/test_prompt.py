"""
提示词工程
"""

from langchain_core.prompts import PromptTemplate

# 制定模板，包括角色，技能，要求
template = """
    你是一个专业的{role}，请根据以下信息：为{company}撰写一份{document_type}。
    要求：
    1. 使用{style}风格
    2. 内容要{requirements}
    
    主题：{topic}
"""

# 创建Prompt模板对象
prompt = PromptTemplate(
    input_variables=["role", "company", "document_type", "style", "requirements", "topic"],
    template=template, 
)

# 填充模板
formatted_prompt = prompt.format(
    role="专业的财务分析师",
    company="白泽科技公司", 
    document_type="年度报告", 
    style="正式商务", 
    requirements="数据详实，分析深入",
    topic="2025年财务表现分析"
)

# 测试结果
print(formatted_prompt)