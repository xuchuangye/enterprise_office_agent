"""
自定义工具：实现计算器功能的工具，支持基本数学运算和表达式计算
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

from typing import Type
from pydantic import BaseModel, Field
import math
import logging
from .base import BaseTool

# 启动日志
logger = logging.getLogger(__name__)

# 定义参数模型
class CalculatorInput(BaseModel):
    """
    计算器输入参数
    """
    expression: str = Field(..., description="要计算的数学表达式， 例如： 2+3+4, sqrt(9), sin(3.14)")

# 定义工具类以及计算规则
class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = """
    一个计算器工具，能够计算数学表达式的结果。
    支持的操作：
    - 基本运算：加(+), 减(-), 乘(*), 除(/)
    - 指数运算：**
    - 括号运算：()
    - 数学函数： sin, cos, tan, sqrt, log, exp
    """

    # 参数校验
    args_schema: Type[BaseModel] = CalculatorInput # type: ignore

    # 工具的执行逻辑
    def _execute(self, expression: str, **kwargs) -> str:
        try:
            # 定义安全的全局变量（只包含允许的函数和常量）
            safe_globals = {
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log10": math.log10,
                "abs": abs,
                "round": round,
                "exp": math.exp,
                "log": math.log,
                "pi": math.pi,
                "e": math.e,
            }

            # 执行表达式计算
            result = eval(expression, safe_globals, {})
            logger.info(f"执行表达式计算的结果: {result}")
            # 对结果进行检查，是否为数字
            if isinstance(result, (int, float)):
                
                # 格式化输出，避免科学计数法
                if abs(result) >= 1e10 or (abs(result) < 1e6 and result != 0):
                    return f"结果：{result}"
                else:
                    if result == int(result):
                        return f"结果：{int(result)}"
                    else:
                        return f"结果：{result}"
            else:
                return "计算结果：{result}"
            
        except Exception as e:
            logging.error(f"计算失败: {e}", exc_info=True)
            return "计算失败"