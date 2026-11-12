"""
工具基类定义
所有自定义工具都继承自 LangChain 的 BaseTool
"""
from langchain_core.tools import BaseTool as LangChainBaseTool
from abc import abstractmethod
from typing import Any, ClassVar, Iterator, Optional, Type
from pydantic import BaseModel

import logging

logger = logging.getLogger(__name__)


class BaseTool(LangChainBaseTool):
    
    """
    工具基类，扩展了 LangChain BaseTool
    
    子类需要实现：
    - name: 工具名称
    - description: 工具描述
    - _execute: 具体的业务逻辑（接收参数）
    - _run: 同步执行方法（已自动处理参数）
    - _arun: 异步执行方法（可选）
    - args_schema: Pydantic 参数模型（可选）
    """
    # 标记该工具的输出是否由 LLM 生成、支持逐 token 流式返回。
    # 只有真正实现 stream_result 的工具才应置为 True
    streams_llm_output: ClassVar[bool] = False

    # 使用Pydantic做参数校验
    # 用户传入错误的类型参数，没有参数校验，可能会导致系统直接崩溃或者产生不可控不可预期的结果
    # 所以Pydantic参数校验的价值
    # 1. 类型安全，确保参数类型是正确的，会自动生成参数说明的文档，提供友好的清晰的错误提示信息，能够将str自动转换为正确的类型
    # 重写 _run，兼容 LangChain 可能传入的 *args 和 **kwargs
    def _run(self, *args: Any, **kwargs: Any) -> str:
        """
        工具调用入口
        LangChain 会根据 args_schema 将参数自动解析为 kwargs键值对，
        但某些情况下也可能传入位置参数，因此这里做了兼容。
        """
        try:
            logger.info(f"执行工具: {self.name}")
            # 如果子类实现了 _execute，将参数全部传递
            return self._execute(*args, **kwargs)
        except Exception as e:
            logger.exception("工具执行出错")
            return f"工具执行出错: {str(e)}"
    
    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """异步执行（默认调用同步方法）"""
        return self._run(*args, **kwargs)

    def stream_result(self, *args: Any, **kwargs: Any) -> Iterator[str]:
        """
        流式执行工具，逐块返回文本。

        默认退化为一次性返回完整结果；输出由 LLM 生成的工具（如知识问答）
        可以重写本方法，用 LLM.stream() 边生成边返回，降低用户等待焦虑。
        """
        yield self._run(*args, **kwargs)

    @abstractmethod # 强制子类实现
    def _execute(self, *args, **kwargs) -> str:
        """子类实现具体的工具逻辑，需要子类重写"""
        pass