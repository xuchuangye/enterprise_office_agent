import json
import unittest
from unittest.mock import patch

from agent_core.memory_manager import MemoryManager


class FakeRedis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        value = self.values.get(key)
        return json.loads(value) if value else None

    def set(self, key, value):
        self.values[key] = json.dumps(value, ensure_ascii=False)
        return True


class FakeMySQL:
    def __init__(self, messages):
        self.messages = messages
        self.preferences = {}

    def get_preference(self, user_id, key):
        return self.preferences.get((user_id, key))

    def set_preference(self, user_id, key, value):
        self.preferences[(user_id, key)] = value
        return True

    def get_conversation_window(self, session_id, limit):
        return self.messages[-limit:]

    def get_conversation_batch(self, session_id, after_id, before_id, limit):
        return [
            message for message in self.messages
            if after_id < message["id"] < before_id
        ][:limit]


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = responses or ["合并后的事实摘要"]
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        return response


def build_manager(message_count):
    manager = MemoryManager.__new__(MemoryManager)
    manager.redis = FakeRedis()
    manager.mysql = FakeMySQL([
        {
            "id": index,
            "role": "user" if index % 2 else "assistant",
            "content": f"对话消息{index}",
        }
        for index in range(1, message_count + 1)
    ])
    manager._short_term_cache = {}
    return manager


class MemorySummaryTests(unittest.TestCase):
    def test_summary_rolls_forward_in_bounded_batches(self):
        manager = build_manager(15)
        llm = FakeLLM([
            "第一批摘要",
            "第二批摘要",
            "最终摘要",
        ])

        with (
            patch("agent_core.memory_manager.settings.SHORT_TERM_MEMORY_MAX_LENGTH", 10),
            patch("agent_core.memory_manager.settings.MEMORY_SUMMARY_BATCH_SIZE", 2),
        ):
            summary = manager.update_summary("session-1", llm)

        self.assertEqual(summary, "最终摘要")
        self.assertEqual(len(llm.prompts), 3)
        self.assertIn("对话消息1", llm.prompts[0])
        self.assertIn("第一批摘要", llm.prompts[1])
        self.assertIn("对话消息5", llm.prompts[2])
        state = json.loads(
            manager.mysql.preferences[("session:session-1", "summary_state")]
        )
        self.assertEqual(state, {"summary": "最终摘要", "last_message_id": 5})

    def test_empty_model_response_does_not_advance_cursor(self):
        manager = build_manager(15)
        llm = FakeLLM(["  "])

        with patch("agent_core.memory_manager.settings.SHORT_TERM_MEMORY_MAX_LENGTH", 10):
            summary = manager.update_summary("session-1", llm)

        self.assertEqual(summary, "")
        self.assertNotIn(("session:session-1", "summary_state"), manager.mysql.preferences)

    def test_summary_prompt_limits_each_message_and_context(self):
        manager = build_manager(11)
        manager.mysql.messages[0]["content"] = "x" * 100
        with (
            patch("agent_core.memory_manager.settings.SHORT_TERM_MEMORY_MAX_LENGTH", 10),
            patch("agent_core.memory_manager.settings.MEMORY_SUMMARY_MESSAGE_MAX_CHARS", 20),
            patch("agent_core.memory_manager.settings.MEMORY_CONTEXT_MAX_CHARS", 40),
        ):
            prompt = manager._build_summary_prompt("", manager.mysql.messages[:2])

        self.assertLessEqual(prompt.count("x"), 20)
        self.assertIn("该消息已截断", prompt)


if __name__ == "__main__":
    unittest.main()
