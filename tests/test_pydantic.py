"""
测试Python数据的参数校验Pydantic
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

from tools.data_analyzer import DataAnalyzerInput

# 正确调用
input_data = DataAnalyzerInput(
    action="read_data", 
    file_path="data/sales_data.xlsx", 
)

# 错误调用
try:
    input_data = DataAnalyzerInput(
        action="", 
        file_path="", 
        # auto_generate_doc
        # Input should be a valid boolean, unable to interpret input [type=bool_parsing, input_value='', input_type=str]
        auto_generate_doc=""
    )
except Exception as e:
    print(f"参数校验错误: {str(e)}")