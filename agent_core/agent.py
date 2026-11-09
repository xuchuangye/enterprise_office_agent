"""
核心Agent封装
基于LangChain ReAct Agent，集成记忆、工具调用、反思功能
支持复杂任务的自动规划与多步骤执行
"""
import os
import uuid
from typing import Optional, List, Dict, Any
import logging

# LangChain 1.0+ 版本对 Agents 模块进行了重大重构，AgentExecutor 和 create_react_agent 已从主包 langchain 中正式移除
# from langchain.agents import AgentExecutor, create_react_agent
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate
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

logger = logging.getLogger(__name__)


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
    def _create_agent_executor(self, session_id: str) -> AgentExecutor:
        # 注册工具
        tools = self.tool_registry.list_tools()
        if not tools:
            raise ValueError("没有注册任何工具，请先注册工具")
        # 提示词模板
        prompt = PromptTemplate.from_template(self.REACT_TEMPLATE)
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
        agent = create_react_agent(llm=self.llm, tools=tools, prompt=prompt) # type: ignore
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

    # 复杂任务使用任务规划器
    def _execute_planned_task(self, user_input: str, session_id: str,
                               user_id: Optional[str] = None) -> Dict[str, Any]:
        """多步骤任务执行（使用任务规划器）---双路径的执行策略"""
        available_tools = self.tool_registry.get_tool_names()
        task_steps = self.task_planner.plan(user_input, available_tools)
        if not task_steps:
            # ReAct适合执行简单任务
            return self._execute_single_step(user_input, session_id, user_id)

        logger.info(f"任务规划成功，共 {len(task_steps)} 个步骤")
        all_steps = []
        final_output = ""

        for i, step in enumerate(task_steps):
            tool_name = step.get("tool")
            params = step.get("params", {})
            logger.info(f"执行步骤 {i+1}/{len(task_steps)}: {tool_name}")

            tool = self.tool_registry.get_tool(tool_name) # type: ignore
            if not tool:
                error_msg = f"错误：未找到工具 {tool_name}"
                all_steps.append({"step": i+1, "tool": tool_name, "error": error_msg})
                final_output += f"\n[步骤{i+1} 失败] {error_msg}"
                continue

            try:
                # 修复：使用标准调用 tool.invoke(params)
                observation = tool.invoke(params)
                all_steps.append({
                    "tool": tool_name,
                    "tool_input": params,
                    "observation": observation
                })
                final_output += f"\n[步骤{i+1} 完成] {tool_name}: {observation[:200]}"
                self.memory_manager.mysql.log_tool_call(
                    session_id=session_id,
                    user_id=user_id or "anonymous",
                    tool_name=tool_name, # type: ignore
                    params=params,
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
                             user_id: Optional[str] = None) -> Dict[str, Any]:
        """单步任务（ReAct Agent）"""
        try:
            # 创建Agent执行器
            executor = self._create_agent_executor(session_id)
            user_context = ""
            if user_id:
                prefs = self.memory_manager.get_all_preferences(user_id)
                if prefs:
                    user_context = f"\n用户偏好信息：{prefs}"
            # 执行Agent，获取结果和中间步骤
            result = executor.invoke({"input": user_input + user_context})
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
    def run(self, user_input: str, session_id: Optional[str] = None,
            user_id: Optional[str] = None, use_planner: Optional[bool] = None) -> Dict[str, Any]:
        if session_id is None:
            session_id = str(uuid.uuid4())
        if not self.llm:
            return {
                "output": "LLM未初始化，请配置API Key后重试。",
                "intermediate_steps": [],
                "session_id": session_id
            }
        if use_planner is None:
            use_planner = self.enable_planner and self._is_complex_task(user_input)
        if use_planner:
            logger.info("使用任务规划器处理请求")
            return self._execute_planned_task(user_input, session_id, user_id)
        else:
            logger.info("使用ReAct Agent处理请求")
            return self._execute_single_step(user_input, session_id, user_id)

    def clear_session(self, session_id: str):
        self.memory_manager.clear_session(session_id)