"""
记忆管理器
统一管理短期记忆（会话上下文）和长期记忆（用户偏好）
支持Redis持久化，失败时降级为内存存储
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
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

from storage import RedisClient, MySQLClient
from config.settings import settings

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self):
        # Redis存储
        self.redis = RedisClient()
        # MySQL存储
        self.mysql = MySQLClient()
        # 本地缓存列表
        self._short_term_cache: Dict[str, List[Dict[str, Any]]] = {}

    def add_message(self, session_id: str, role: str, content: str) -> None:
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        key = f"chat:{session_id}"

        # Redis 存储
        self.redis.lpush(key, message)
        self.redis.ltrim(key, 0, settings.SHORT_TERM_MEMORY_MAX_LENGTH - 1)

        # MySQL 持久化
        self.mysql.add_conversation(session_id, role, content)

        # 降级缓存：无论 Redis 是否可用，都维护本地缓存列表
        if session_id not in self._short_term_cache:
            self._short_term_cache[session_id] = []
        self._short_term_cache[session_id].append(message)
        if len(self._short_term_cache[session_id]) > settings.SHORT_TERM_MEMORY_MAX_LENGTH:
            self._short_term_cache[session_id] = self._short_term_cache[session_id][-settings.SHORT_TERM_MEMORY_MAX_LENGTH:]

    def get_messages(self, session_id: str, limit: int = 0) -> List[Dict[str, Any]]:
        """
        获取会话消息
        """
        key = f"chat:{session_id}"
        messages = []
        # 如果Redis是有效的，那么就从Redis中获取
        if self.redis.available:
            raw_msgs = self.redis.lrange(key, 0, -1)
            messages = list(reversed(raw_msgs))
        # 如果Redis是无效的，那么就从本地缓存列表中获取
        elif session_id in self._short_term_cache:
            messages = self._short_term_cache[session_id].copy()
        # 如果本地缓存列表中也没有，那么就从MySQL中获取，并且添加到本地缓存列表
        else:
            # 从 MySQL 获取并加载到缓存
            db_msgs = self.mysql.get_conversation_history(session_id, limit or 20)
            for msg in db_msgs:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["created_at"].isoformat() if hasattr(msg["created_at"], "isoformat") else str(msg["created_at"])
                })
            # 添加到本地缓存列表
            self._short_term_cache[session_id] = messages.copy()

        if limit:
            messages = messages[-limit:]
        return messages

    def get_chat_history_string(self, session_id: str, limit: int = 6) -> str:
        """
        获取会话历史记录
        """
        messages = self.get_messages(session_id, limit)
        if not messages:
            return "（无历史对话）"
        lines = [f"{'用户' if m['role']=='user' else '助手'}: {m['content']}" for m in messages]
        return "\n".join(lines)

    def clear_session(self, session_id: str) -> None:
        """
        清除会话记忆
        """
        key = f"chat:{session_id}"
        self.redis.delete(key)
        self._short_term_cache.pop(session_id, None)
        logger.info(f"会话 {session_id} 记忆已清除")
    
    # ========== 长期记忆（用户偏好）==========
    def set_preference(self, user_id: str, key: str, value: Any) -> bool:
        """
        设置用户偏好
        
        Args:
            user_id: 用户ID
            key: 偏好键
            value: 偏好值
        
        Returns:
            是否成功
        """
        # 存入Redis
        redis_key = f"{settings.LONG_TERM_MEMORY_PREFIX}{user_id}:{key}"
        self.redis.set(redis_key, value)
        # 存入MySQL
        return self.mysql.set_preference(user_id, key, json.dumps(value, ensure_ascii=False))
    
    def get_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        获取用户偏好
        
        Args:
            user_id: 用户ID
            key: 偏好键
            default: 默认值
        
        Returns:
            偏好值
        """
        # 优先从Redis获取
        redis_key = f"{settings.LONG_TERM_MEMORY_PREFIX}{user_id}:{key}"
        value = self.redis.get(redis_key)
        if value is not None:
            return value
        
        # 从MySQL获取
        db_value = self.mysql.get_preference(user_id, key)
        if db_value is not None:
            try:
                return json.loads(db_value)
            except:
                return db_value
        
        return default
    
    def get_all_preferences(self, user_id: str) -> Dict[str, Any]:
        """获取用户所有偏好"""
        prefs = self.mysql.get_all_preferences(user_id)
        result = {}
        for k, v in prefs.items():
            try:
                result[k] = json.loads(v)
            except:
                result[k] = v
        return result
    
    # ========== 记忆摘要（用于上下文压缩）==========
    @staticmethod
    def _truncate_message(content: str, max_chars: int) -> str:
        """截断单条消息，避免超长内容挤占摘要上下文"""
        if len(content) <= max_chars:
            return content
        return content[:max_chars] + "…（该消息已截断）"

    def _build_summary_prompt(self, previous_summary: str, messages: List[Dict[str, Any]]) -> str:
        """构建滚动摘要提示词，同时限制单条消息和整体上下文的长度"""
        max_context_chars = settings.MEMORY_CONTEXT_MAX_CHARS
        lines = []
        used_chars = 0
        for message in messages:
            content = self._truncate_message(
                str(message.get("content", "")),
                settings.MEMORY_SUMMARY_MESSAGE_MAX_CHARS,
            )
            role = "用户" if message.get("role") == "user" else "助手"
            line = f"{role}: {content}"
            # 至少保留一条，超出的较早内容直接省略，控制 prompt 体积
            if lines and used_chars + len(line) + 1 > max_context_chars:
                lines.append("（更早内容已省略）")
                break
            lines.append(line)
            used_chars += len(line) + 1
        conversation = "\n".join(lines) or "（无）"

        previous = (previous_summary or "").strip()
        if len(previous) > settings.MEMORY_SUMMARY_MAX_CHARS:
            previous = previous[:settings.MEMORY_SUMMARY_MAX_CHARS] + "…（摘要已截断）"

        return (
            "你是一个会话记忆压缩助手。请把【已有摘要】与【新对话】合并成一段简洁、"
            "事实准确的摘要，保留关键信息、用户需求、已确认的结论以及未完成事项，"
            "不要编造对话中没有出现的内容。\n\n"
            f"【已有摘要】\n{previous or '（无）'}\n\n"
            f"【新对话】\n{conversation}\n\n"
            "请只输出合并后的摘要正文："
        )

    def update_summary(self, session_id: str, llm=None) -> str:
        """
        滚动更新会话摘要（上下文压缩）

        只摘要“短期窗口之外”的历史消息，并按 MEMORY_SUMMARY_BATCH_SIZE 分批调用 LLM，
        用 user_preferences 里的 summary_state 记录 (summary, last_message_id) 游标。
        LLM 返回空内容时不推进游标，避免丢失未摘要的对话。

        Args:
            session_id: 会话ID（不带 chat: 前缀）
            llm: LLM实例

        Returns:
            当前摘要；没有可摘要内容或失败时返回原摘要/空串
        """
        if not llm:
            return ""

        state = self.get_preference(f"session:{session_id}", "summary_state")
        if isinstance(state, str):
            try:
                state = json.loads(state)
            except (TypeError, ValueError):
                state = {}
        if not isinstance(state, dict):
            state = {}
        # 获取上次摘要和游标
        previous_summary = str(state.get("summary") or "")
        try:
            last_message_id = int(state.get("last_message_id") or 0)
        except (TypeError, ValueError):
            last_message_id = 0

        # 短期窗口内的消息保持原样，只压缩窗口之外的部分
        window = self.mysql.get_conversation_window(
            session_id, settings.SHORT_TERM_MEMORY_MAX_LENGTH
        ) or []
        if not window:
            return previous_summary
        cutoff_id = int(window[0]["id"]) - 1
        if cutoff_id <= last_message_id:
            return previous_summary

        summary = previous_summary
        while last_message_id < cutoff_id:
            batch = self.mysql.get_conversation_batch(
                session_id,
                last_message_id,
                cutoff_id + 1,
                settings.MEMORY_SUMMARY_BATCH_SIZE,
            ) or []
            if not batch:
                break

            prompt = self._build_summary_prompt(summary, batch)
            try:
                response = llm.invoke(prompt)
            except Exception as e:
                logger.exception(f"生成对话摘要失败: {e}")
                return ""

            response = str(response or "").strip()
            if not response:
                logger.warning("LLM 返回空摘要，保留原摘要并且不推进游标")
                return ""

            summary = response
            last_message_id = int(batch[-1]["id"])
        # 如果新的摘要的游标与原摘要的游标不同，说明有新的内容被摘要了，更新 summary_state
        # 同时记录当前的session_id和last_message_id，方便下次摘要时从上次的游标继续
        if summary != previous_summary:
            self.set_preference(
                f"session:{session_id}",
                "summary_state",
                {"summary": summary, "last_message_id": last_message_id},
            )
        return summary

    def generate_summary(self, session_id: str, llm=None) -> str:
        """
        生成会话摘要（调用LLM总结对话）
        
        Args:
            session_id: 会话ID
            llm: LLM实例
        
        Returns:
            摘要字符串
        """
        messages = self.get_messages(session_id, limit=20)  # 获取最近20条消息，包括用户和助手的对话
        if not messages or not llm:
            return "会话暂无足够内容"
        
        conversation = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt = f"""
            请对以下对话进行简洁的摘要总结，突出关键信息和用户需求。
            对话内容：
            {conversation}

            摘要："""
        
        try:
            summary = llm.invoke(prompt)
            self.set_preference(f"session:{session_id}", "summary", summary)
            return summary.strip()
        except Exception as e:
            logger.exception(f"生成摘要失败: {e}")
            return ""

# 单例初始化MemoryManager
import os
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()
# 获取API Key
api_key = os.getenv("DASHSCOPE_API_KEY")

from langchain_community.llms import Tongyi
llm = Tongyi(model_name=settings.LLM_MODEL_NAME, api_key=api_key)
if __name__ == "__main__":
    memoryManager = MemoryManager()
    # 支持 python agent_core/memory_manager.py <session_id>，session_id 不带 chat: 前缀
    session_id = sys.argv[1] if len(sys.argv) > 1 else "test_001"
    print(memoryManager.update_summary(session_id, llm=llm))  # 测试滚动生成摘要