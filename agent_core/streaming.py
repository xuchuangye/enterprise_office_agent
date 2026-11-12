"""
流式输出支持

LangChain 的 AgentExecutor.invoke 本身是阻塞调用，但 ReAct 代理内部通过
RunnableAgent.plan -> LLM.stream 获取模型输出，因此挂上回调处理器后，
即使在阻塞调用期间也能收到逐 token 的 on_llm_new_token 事件。

这里把这些底层回调转换成前端可统一消费的 dict 事件：
- status: 执行进度（规划、调用工具等），用于缓解等待焦虑
- token:  LLM 生成的可见文本增量

注意：ReAct 模板的输出包含 Thought/Action 等中间推理内容，
只有 "Final Answer:" 之后的部分才是给用户看的回答，需要在回调里过滤。
"""
from typing import Any, Callable, Dict

from langchain_core.callbacks import BaseCallbackHandler

# 事件接收函数（Streamlit 直接渲染，FastAPI 写入 SSE 队列）
EventSink = Callable[[Dict[str, Any]], None]

# ReAct 模板要求的最终答案标记。不同模型可能输出中英文或全角冒号，这里全部识别
_FINAL_ANSWER_MARKERS = (
    "Final Answer:",
    "Final Answer：",
    "最终答案:",
    "最终答案：",
)


class StreamingCallbackHandler(BaseCallbackHandler):
    """把 LLM token 和工具调用事件转换成前端可消费的流式事件"""

    def __init__(self, sink: EventSink) -> None:
        super().__init__()
        self._sink = sink
        self._buffer = ""
        self._answer_started = False

    # 每一轮新的 LLM 调用（新的 Thought/Action）都要重置过滤状态
    def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
        self._buffer = ""
        self._answer_started = False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if not token:
            return
        # 已经开始输出最终答案，后续 token 直接透传
        if self._answer_started:
            self._sink({"type": "token", "content": token})
            return

        # 标记可能被拆到多个 token 里（如 "Final " + "Answer:"），先累积再查找
        self._buffer += token
        for marker in _FINAL_ANSWER_MARKERS:
            index = self._buffer.find(marker)
            if index == -1:
                continue
            self._answer_started = True
            content = self._buffer[index + len(marker):]
            if content:
                self._sink({"type": "token", "content": content})
            return

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        """ReAct 选择工具时给出进度提示"""
        tool_name = getattr(action, "tool", "工具")
        self._sink({"type": "status", "message": f"正在调用工具 {tool_name}…"})

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self._sink({"type": "status", "message": "正在分析工具返回结果…"})
