"""
核心Agent封装
基于LangChain ReAct Agent，集成记忆、工具调用、反思功能
支持复杂任务的自动规划与多步骤执行
"""
import os
import queue
import threading
import uuid
from typing import Optional, List, Dict, Any, Iterator
import logging

# LangChain 1.0+ 版本对 Agents 模块进行了重大重构，AgentExecutor 和 create_react_agent 已从主包 langchain 中正式移除
# from langchain.agents import AgentExecutor, create_react_agent
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.tools.render import render_text_description_and_args
# 新版 langchain 包已移除经典 Memory 模块。
# from langchain.memory import ConversationBufferWindowMemory
# langchain-community 不包含 ConversationBufferWindowMemory（它只提供第三方集成，如数据库连接器）
# from langchain_community.memory import ConversationBufferWindowMemory
from langchain_classic.memory import ConversationBufferWindowMemory


from config.settings import settings
from .memory_manager import MemoryManager
from .tool_registry import ToolRegistry
from .reflection import Reflection
from .planner import TaskPlanner
from .streaming import StreamingCallbackHandler, EventSink

logger = logging.getLogger(__name__)

# stream() 的结束哨兵：worker 线程执行完毕后放入队列，主线程据此结束迭代
_STREAM_END = object()


class CustomConversationMemory(ConversationBufferWindowMemory):
    """扩展记忆类，在保存上下文的时候同步保存到 MemoryManager，并且避免多 key 警告"""
    memory_manager: Optional[MemoryManager] = None
    session_id: str = ""

    class Config:
        arbitrary_types_allowed = True

    # 重写 save_context 方法，同步到 MemoryManager
    def save_context(self, inputs: dict, outputs: dict) -> None:
        output_text = outputs.get("output", "")
        clean_outputs = {"output": output_text}
        # 调用父类的方法，保持LangChain原有的记忆功能
        super().save_context(inputs, clean_outputs)

        if self.memory_manager:
            user_input = inputs.get("input", "")
            # 同步用户的输入，将用户输入保存到记忆管理器当中
            if user_input:
                self.memory_manager.add_message(self.session_id, "user", user_input)
            # 同步助手的回答，将助手的回答也同步保存到记忆管理器中
            if output_text:
                self.memory_manager.add_message(self.session_id, "assistant", output_text)


class OfficeAgent:
    # ReAct 模板
    REACT_TEMPLATE = """
    你是一个企业智能办公助手，能够使用各种工具帮助用户完成办公任务。

    你可以使用的工具有：
    {tools}

    工具名称列表：{tool_names}

    请严格按照以下格式回答：

    Question: 用户的问题
    Thought: 思考需要做什么
    Action: 要使用的工具名称（必须是 [{tool_names}] 中的一个）
    Action Input: 工具的输入参数（JSON格式）
    Observation: 工具返回的结果
    ... (这个 Thought/Action/Action Input/Observation 可以重复多次)
    Thought: 我现在知道最终答案了
    Final Answer: 对用户的最终回复

    开始！

    历史对话摘要（较早对话的压缩结果，仅作为背景事实参考）：
    {conversation_summary}

    历史对话：
    {chat_history}

    Question: {input}
    Thought: {agent_scratchpad}
    """

    def __init__(self, llm: Optional[Tongyi] = None, enable_planner: bool = True):
        if llm:
            self.llm = llm
        else:
            self._init_llm()
        # 记忆管理器，存储对话历史
        self.memory_manager = MemoryManager()
        # 工具注册表，管理所有可用的工具
        self.tool_registry = ToolRegistry()
        # 反思模块，分析失败原因并且生成替代策略
        self.reflection = Reflection(self.llm)
        # 任务执行器
        self.agent_executor: Optional[AgentExecutor] = None
        # 任务规划器，用于拆解复杂任务
        self.enable_planner = enable_planner and (self.llm is not None)
        if self.enable_planner:
            self.task_planner = TaskPlanner(self.llm)
        logger.info(f"OfficeAgent初始化完成，Planner模式: {'开启' if self.enable_planner else '关闭'}")

    def _init_llm(self):
        # 配置DashScope的API Key
        if settings.DASHSCOPE_API_KEY:
            os.environ["DASHSCOPE_API_KEY"] = settings.DASHSCOPE_API_KEY
            self.llm = Tongyi(
                # 模型名称
                model_name=settings.LLM_MODEL_NAME,
                # 模型温度参数
                temperature=settings.LLM_TEMPERATURE,
                # 最大生成长度
                max_tokens=settings.LLM_MAX_TOKENS
            )
            logger.info(f"LLM初始化成功，模型: {settings.LLM_MODEL_NAME}")
        else:
            logger.warning("未配置API Key，Agent将无法正常工作")
            self.llm = None

    # Agent集成单个工具
    def register_tool(self, tool):
        self.tool_registry.register(tool)

    # Agent集成批量工具
    def register_tools(self, tools: List):
        self.tool_registry.register_many(tools)

    # 创建Agent执行器
    def _create_agent_executor(self, session_id: str,
                               conversation_summary: str = "") -> AgentExecutor:
        # 注册工具
        tools = self.tool_registry.list_tools()
        if not tools:
            raise ValueError("没有注册任何工具，请先注册工具")
        # 提示词模板，提前填充滚动摘要，避免模板变量缺失
        prompt = PromptTemplate.from_template(self.REACT_TEMPLATE).partial(
            conversation_summary=conversation_summary or "（无）"
        )
        # 创建记忆对象，加载历史对话
        memory = CustomConversationMemory(
            memory_key="chat_history",
            return_messages=False,
            input_key="input",
            k=settings.SHORT_TERM_MEMORY_MAX_LENGTH,
            memory_manager=self.memory_manager,
            session_id=session_id
        )
        # 获取所有的历史对话
        history = self.memory_manager.get_messages(session_id)
        for msg in history:
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            else:
                memory.chat_memory.add_ai_message(msg["content"])
        # 创建React代理的Agent
        # 使用 render_text_description_and_args，让 {tools} 中同时包含参数 Schema，
        # 否则 ReAct 路径同样容易编造参数名
        agent = create_react_agent(
            llm=self.llm, tools=tools, prompt=prompt,
            tools_renderer=render_text_description_and_args
        ) # type: ignore
        # 创建执行器
        executor = AgentExecutor(
            agent=agent, tools=tools, memory=memory,
            verbose=settings.DEBUG,                     # 调试模式
            handle_parsing_errors=True,                 # 忽略解析错误
            # max_iterations=8表示最多执行8次，超过8次就停止执行，防止死循环
            max_iterations=8, return_intermediate_steps=True
        )
        return executor

    # 复杂任务的判断
    def _is_complex_task(self, user_input: str) -> bool:
        """通过关键词判断任务是否复杂，包含这些关键词就需要使用规划器"""
        complex_keywords = ["分析", "报告", "生成", "图表", "可视化", "工作总结", "会议纪要", "审批"]
        return any(kw in user_input for kw in complex_keywords)

    @staticmethod
    def _prepare_tool_params(tool: Any, params: Any) -> Dict[str, Any]:
        """
        执行前按工具的 args_schema 再校验/规范化一次参数。
        规划器已经校验过 LLM 生成的步骤，这里是最后一道防线，
        防止错误的字段名（如 query 而不是 question）直接传给工具。
        """
        schema = getattr(tool, "args_schema", None)
        if schema is None:
            return params
        if not isinstance(params, dict):
            raise ValueError(
                f"工具 {tool.name} 的参数必须是 JSON 对象，收到: {type(params).__name__}"
            )
        validated = schema.model_validate(params)
        return validated.model_dump(exclude_none=True)

    # 复杂任务使用任务规划器
    def _execute_planned_task(self, user_input: str, session_id: str,
                               user_id: Optional[str] = None,
                               conversation_summary: str = "",
                               event_sink: Optional[EventSink] = None) -> Dict[str, Any]:
        """多步骤任务执行（使用任务规划器）---双路径的执行策略"""
        # 传入工具实例而不是名称，让规划器能拿到描述和参数 Schema
        available_tools = self.tool_registry.list_tools()
        if event_sink:
            event_sink({"type": "status", "message": "正在规划任务步骤…"})
        task_steps = self.task_planner.plan(
            user_input, available_tools, conversation_summary=conversation_summary
        )
        if not task_steps:
            # ReAct适合执行简单任务
            return self._execute_single_step(
                user_input, session_id, user_id,
                conversation_summary=conversation_summary,
                event_sink=event_sink
            )

        logger.info(f"任务规划成功，共 {len(task_steps)} 个步骤")
        all_steps = []
        final_output = ""

        for i, step in enumerate(task_steps):
            tool_name = step.get("tool")
            params = step.get("params", {})
            logger.info(f"执行步骤 {i+1}/{len(task_steps)}: {tool_name}")
            if event_sink:
                event_sink({
                    "type": "status",
                    "message": f"正在执行步骤 {i+1}/{len(task_steps)}：{tool_name}…"
                })

            tool = self.tool_registry.get_tool(tool_name) # type: ignore
            if not tool:
                error_msg = f"错误：未找到工具 {tool_name}"
                all_steps.append({"step": i+1, "tool": tool_name, "error": error_msg})
                final_output += f"\n[步骤{i+1} 失败] {error_msg}"
                continue

            try:
                # 用 args_schema 规范化参数后再调用，避免字段名不匹配
                tool_params = self._prepare_tool_params(tool, params)
                if event_sink and getattr(tool, "streams_llm_output", False):
                    # LLM 生成输出的工具（如知识问答）一边生成一边推给前端
                    chunks: List[str] = []
                    for chunk in tool.stream_result(**tool_params): # type: ignore[attr-defined]
                        chunks.append(chunk)
                        event_sink({"type": "token", "content": chunk})
                    observation = "".join(chunks)
                else:
                    # 修复：使用标准调用 tool.invoke(params)
                    observation = tool.invoke(tool_params)
                all_steps.append({
                    "tool": tool_name,
                    "tool_input": tool_params,
                    "observation": observation
                })
                # 保留完整结果，避免知识问答等内容被截断
                final_output += f"\n[步骤{i+1} 完成] {tool_name}: {observation}"
                self.memory_manager.mysql.log_tool_call(
                    session_id=session_id,
                    user_id=user_id or "anonymous",
                    tool_name=tool_name, # type: ignore
                    params=tool_params,
                    result=str(observation)[:1000]
                )
            except Exception as e:
                error_msg = f"执行出错: {str(e)}"
                logger.error(f"步骤 {i+1} 执行失败: {e}")
                all_steps.append({"step": i+1, "tool": tool_name, "error": error_msg})
                final_output += f"\n[步骤{i+1} 出错] {error_msg}"

        return {
            "output": final_output.strip() or "任务执行完成",
            "intermediate_steps": all_steps,
            "session_id": session_id,
            "planned": True,
            "steps_count": len(task_steps)
        }
    # 任务简单使用ReAct
    def _execute_single_step(self, user_input: str, session_id: str,
                             user_id: Optional[str] = None,
                             conversation_summary: str = "",
                             event_sink: Optional[EventSink] = None) -> Dict[str, Any]:
        """单步任务（ReAct Agent）"""
        try:
            # 创建Agent执行器
            executor = self._create_agent_executor(session_id, conversation_summary)
            user_context = ""
            if user_id:
                prefs = self.memory_manager.get_all_preferences(user_id)
                if prefs:
                    user_context = f"\n用户偏好信息：{prefs}"
            if event_sink:
                event_sink({"type": "status", "message": "正在思考并选择工具…"})
            # ReAct 代理内部走 LLM.stream，挂回调即可在阻塞调用期间拿到 token
            config = None
            if event_sink:
                config = {"callbacks": [StreamingCallbackHandler(event_sink)]}
            # 执行Agent，获取结果和中间步骤
            result = executor.invoke({"input": user_input + user_context}, config=config)
            steps = result.get("intermediate_steps", [])
            for step in steps:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    self.memory_manager.mysql.log_tool_call(
                        session_id=session_id,
                        user_id=user_id or "anonymous",
                        tool_name=action.tool,
                        params=action.tool_input,
                        result=str(observation)[:1000]
                    )
            return {
                "output": result.get("output", "执行完成"),
                "intermediate_steps": steps,
                "session_id": session_id,
                "planned": False
            }
        except Exception as e:
            logger.error(f"单步执行失败: {e}", exc_info=True)
            return {
                "output": f"执行失败: {str(e)}",
                "intermediate_steps": [],
                "session_id": session_id,
                "error": str(e)
            }
    # Agent的主入口
    def _run_internal(self, user_input: str, session_id: str,
                      user_id: Optional[str] = None,
                      use_planner: Optional[bool] = None,
                      event_sink: Optional[EventSink] = None) -> Dict[str, Any]:
        """同步执行主流程；event_sink 不为空时额外推送进度和 token 事件"""
        if not self.llm:
            return {
                "output": "LLM未初始化，请配置API Key后重试。",
                "intermediate_steps": [],
                "session_id": session_id
            }
        # 先滚动更新摘要：只压缩短期窗口之外的历史，无新内容时不会调用 LLM
        if event_sink:
            event_sink({"type": "status", "message": "正在整理历史对话…"})
        conversation_summary = ""
        try:
            conversation_summary = self.memory_manager.update_summary(
                session_id, self.llm
            ) or ""
        except Exception as e:
            logger.exception(f"更新对话摘要失败，继续执行: {e}")

        if use_planner is None:
            use_planner = self.enable_planner and self._is_complex_task(user_input)
        if use_planner:
            logger.info("使用任务规划器处理请求")
            return self._execute_planned_task(
                user_input, session_id, user_id,
                conversation_summary=conversation_summary,
                event_sink=event_sink
            )
        else:
            logger.info("使用ReAct Agent处理请求")
            return self._execute_single_step(
                user_input, session_id, user_id,
                conversation_summary=conversation_summary,
                event_sink=event_sink
            )

    def run(self, user_input: str, session_id: Optional[str] = None,
            user_id: Optional[str] = None, use_planner: Optional[bool] = None) -> Dict[str, Any]:
        if session_id is None:
            session_id = str(uuid.uuid4())
        return self._run_internal(user_input, session_id, user_id, use_planner)

    @staticmethod
    def _serialize_steps(steps: Any) -> List[Dict[str, Any]]:
        """把 ReAct 元组 / 规划器字典统一成可 JSON 序列化的步骤结构"""
        serialized: List[Dict[str, Any]] = []
        for step in steps or []:
            if isinstance(step, dict):
                serialized.append({
                    "tool": step.get("tool", "未知工具"),
                    "tool_input": step.get("tool_input") or step.get("params") or {},
                    "observation": str(step.get("observation", step.get("error", ""))),
                })
            elif isinstance(step, (tuple, list)) and len(step) >= 2:
                action, observation = step[0], step[1]
                tool_input = getattr(action, "tool_input", "")
                serialized.append({
                    "tool": getattr(action, "tool", "未知工具"),
                    "tool_input": tool_input if isinstance(tool_input, dict) else {"input": str(tool_input)},
                    "observation": str(observation),
                })
        return serialized

    def stream(self, user_input: str, session_id: Optional[str] = None,
               user_id: Optional[str] = None,
               use_planner: Optional[bool] = None) -> Iterator[Dict[str, Any]]:
        """
        流式执行，逐事件产出，供 Streamlit / FastAPI 消费：

        - {"type": "status", "message": ...}  执行进度
        - {"type": "token", "content": ...}   LLM 生成文本的增量
        - {"type": "result", "output": ..., "steps": [...], "planned": bool, "session_id": ...}
        - {"type": "error", "message": ...}   执行异常

        AgentExecutor.invoke 是阻塞调用，这里放到后台线程执行，
        通过线程安全队列把事件转交给调用方迭代。
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        events: "queue.Queue[Any]" = queue.Queue()

        def sink(event: Dict[str, Any]) -> None:
            events.put(event)

        def worker() -> None:
            try:
                result = self._run_internal(
                    user_input, session_id, user_id, use_planner, event_sink=sink
                )
                sink({
                    "type": "result",
                    "output": result.get("output", ""),
                    "session_id": result.get("session_id", session_id),
                    "planned": bool(result.get("planned")),
                    "steps": self._serialize_steps(result.get("intermediate_steps")),
                })
            except Exception as e:
                logger.exception("流式执行失败")
                sink({"type": "error", "message": str(e), "session_id": session_id})
            finally:
                events.put(_STREAM_END)

        threading.Thread(target=worker, name="agent-stream", daemon=True).start()

        while True:
            event = events.get()
            if event is _STREAM_END:
                break
            yield event

    def clear_session(self, session_id: str):
        self.memory_manager.clear_session(session_id)