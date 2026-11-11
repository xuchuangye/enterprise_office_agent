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
    def generate_summary(self, session_id: str, llm=None) -> str:
        """
        生成会话摘要（调用LLM总结对话）
        
        Args:
            session_id: 会话ID
            llm: LLM实例
        
        Returns:
            摘要字符串
        """
        messages = self.get_messages(session_id, limit=20)
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

from langchain_openai import OpenAI
llm = OpenAI(model="qwen-plus", api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
if __name__ == "__main__":
    memoryManager = MemoryManager()
    print(memoryManager.generate_summary("chat:6f987630-2973-41d5-b775-5b27ce47be01", llm=llm))  # 测试生成摘要