"""
自定义工具：实现计算器的功能
"""
import sys
from pathlib import Path
# 1. 获取当前文件(test_agent.py)的绝对路径
current_file = Path(__file__).resolve()
# 2. 向上回退两级目录: tests -> enterprise_office_agent (项目根目录)
project_root = current_file.parent.parent
# 3. 将项目根目录插入到系统路径的最前面(索引0)
# 这确保了Python会优先在你的项目中查找模块，而不是去site-packages里找
sys.path.insert(0, str(project_root))

import math
from tools.calculator import CalculatorTool

# 创建工具实例
calculator = CalculatorTool()

# 测试用例
test_cases = [
    ("2+3", 5),
    ("3*4", 12),
    ("5-2", 3),
    ("sqrt(9)", 3),
    ("sin(pi/2)", 1),
    ("cos(pi)", -1),
    ("tan(pi)", 0),
    ("log(10)", 1),
    ("exp(1)", math.e),
    ("2^3", 8),
]

for expression, expected_result in test_cases:
    result = calculator.invoke({"expression": expression})
    print(f"输入: {expression}, 输出: {result}")
    print("-" * 50)