"""
测试文件路径提取
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


from agent_core.agent import TaskPlanner


planner = TaskPlanner()

if __name__ == "__main__":
    # 直接文件名
    print(planner._extract_file_path("请分析data/uploads/sales_data.xlsx文件"))
    # 中文冒号
    print(planner._extract_file_path("请分析文件路径：data/uploads/sales_data.xlsx"))