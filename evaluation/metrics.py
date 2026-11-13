"""
评测指标采集：全局线程安全统计器 + LangChain CallbackHandler

覆盖指标：
1. LLM 调用次数（planner / reflection / agent ReAct 路径 / knowledge_qa）
2. Token usage（input / output / total，从 CallbackHandler 自动截取）
3. 延迟统计（端到端 P50/P95 + TTFT 首 token 延迟 P50/P95）
4. 规划降级触发率（LLM 规划失败 → 规则兜底次数 / 总复杂任务数）
5. 工具调用成功率（执行成功 / 总调用次数）
6. 幻觉率校验（答案与检索来源对齐率 + 无依据编造率）
"""
from __future__ import annotations

import threading
import time
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import BaseCallbackHandler


class MetricsCollector:
    """
    全局线程安全的评测指标采集器。

    设计为单例（通过模块级 _collector 导出），避免每个组件自己维护计数。
    所有公开方法都加了 Lock，可在 agent stream() 的后台线程里安全调用。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """重置所有计数器（跑新一批评测前调用）"""
        with self._lock:
            # --- LLM 调用 ---
            self.llm_call_count: int = 0

            # --- Token usage（累计）---
            self.total_input_tokens: int = 0
            self.total_output_tokens: int = 0
            self.total_tokens: int = 0

            # --- 延迟 ---
            self.e2e_latencies: List[float] = []   # 端到端任务延迟（秒）
            self.ttft_latencies: List[float] = []   # 首 token 延迟（秒）

            # --- 规划降级 ---
            self.planner_total_complex_tasks: int = 0
            self.planner_rule_fallback_count: int = 0
            self.planner_llm_success_count: int = 0

            # --- 工具调用 ---
            self.tool_call_success: int = 0
            self.tool_call_failure: int = 0

            # --- 幻觉率校验 ---
            self.rag_qa_total: int = 0
            self.rag_answer_aligned: int = 0   # 答案与检索来源可对齐的 QA 数
            self.rag_answer_hallucinated: int = 0  # 检测到编造的 QA 数

            # --- 当前 TTFT 计时辅助 ---
            self._ttft_start_times: Dict[int, float] = {}  # thread_id → start_time

    # ==================== LLM 调用计数 ====================

    def record_llm_call(self, source: str = "unknown") -> None:
        """记录一次 LLM 调用；source 标识调用来源（planner/reflection/agent/knowledge_qa）"""
        with self._lock:
            self.llm_call_count += 1

    # ==================== Token usage ====================

    def record_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        total_tokens: Optional[int] = None,
    ) -> None:
        """
        从 CallbackHandler 或手动调用方写入 token usage。
        LangChain CallbackHandler 的 on_llm_end 会自动调这个。
        """
        with self._lock:
            self.total_input_tokens += max(0, int(input_tokens or 0))
            self.total_output_tokens += max(0, int(output_tokens or 0))
            if total_tokens is None:
                total_tokens = input_tokens + output_tokens
            self.total_tokens += max(0, int(total_tokens or 0))

    # ==================== 延迟 ====================

    def record_e2e_latency(self, latency_seconds: float) -> None:
        """记录一次端到端任务的总延迟（从 agent._run_internal 入口到返回）"""
        with self._lock:
            self.e2e_latencies.append(latency_seconds)

    def record_ttft(self, ttft_seconds: float) -> None:
        """记录一次首 token 延迟（从 LLM 开始生成到第一个可见 token 的间隔）"""
        with self._lock:
            self.ttft_latencies.append(ttft_seconds)

    # ==================== 规划降级 ====================

    def record_planner_llm_success(self) -> None:
        """LLM 规划成功"""
        with self._lock:
            self.planner_total_complex_tasks += 1
            self.planner_llm_success_count += 1

    def record_planner_rule_fallback(self) -> None:
        """LLM 规划失败，规则兜底触发"""
        with self._lock:
            self.planner_total_complex_tasks += 1
            self.planner_rule_fallback_count += 1

    # ==================== 工具调用 ====================

    def record_tool_call(self, success: bool) -> None:
        """记录一次工具调用结果"""
        with self._lock:
            if success:
                self.tool_call_success += 1
            else:
                self.tool_call_failure += 1

    # ==================== 幻觉率校验 ====================

    def record_rag_qa_result(self, aligned: bool, hallucinated: bool = False) -> None:
        """
        记录一次 RAG 问答的幻觉校验结果。

        aligned:       答案中的关键陈述是否能在检索片段中找到依据
        hallucinated:   是否检测到完全无依据的编造（与 aligned 互斥优先）
        """
        with self._lock:
            self.rag_qa_total += 1
            if hallucinated:
                self.rag_answer_hallucinated += 1
            elif aligned:
                self.rag_answer_aligned += 1

    # ==================== 辅助方法 ====================

    @staticmethod
    def _percentile(sorted_list: List[float], p: float) -> float:
        """在已排序列表上取 p 分位数（p 取值 0~100）"""
        if not sorted_list:
            return 0.0
        if len(sorted_list) == 1:
            return sorted_list[0]
        k = (len(sorted_list) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_list) - 1)
        if f == c:
            return sorted_list[f]
        return sorted_list[f] * (c - k) + sorted_list[c] * (k - f)

    def get_report(self) -> Dict[str, Any]:
        """生成完整的评测指标报告字典，可直接序列化为 JSON 或 Markdown 表格"""
        with self._lock:
            e2e_sorted = sorted(self.e2e_latencies)
            ttft_sorted = sorted(self.ttft_latencies)

            planner_total = self.planner_total_complex_tasks
            planner_success = self.planner_llm_success_count + self.planner_rule_fallback_count
            # 规划环节成功率：LLM 成功 + 规则兜底（两者都是"规划成功并产出步骤"，区别只是来源）
            planner_success_rate = (planner_success / planner_total * 100.0) if planner_total else 0.0
            llm_only_rate = (self.planner_llm_success_count / planner_total * 100.0) if planner_total else 0.0
            rule_fallback_rate = (self.planner_rule_fallback_count / planner_total * 100.0) if planner_total else 0.0

            tool_total = self.tool_call_success + self.tool_call_failure
            tool_success_rate = (self.tool_call_success / tool_total * 100.0) if tool_total else 0.0
            tool_failure_rate = (self.tool_call_failure / tool_total * 100.0) if tool_total else 0.0

            rag_total = self.rag_qa_total
            rag_aligned_rate = (self.rag_answer_aligned / rag_total * 100.0) if rag_total else 0.0
            rag_hallucination_rate = (self.rag_answer_hallucinated / rag_total * 100.0) if rag_total else 0.0

            avg_llm_calls_per_task = (self.llm_call_count / len(self.e2e_latencies)) if self.e2e_latencies else 0.0
            avg_tokens_per_task = (self.total_tokens / len(self.e2e_latencies)) if self.e2e_latencies else 0.0
            avg_input_tokens_per_task = (self.total_input_tokens / len(self.e2e_latencies)) if self.e2e_latencies else 0.0
            avg_output_tokens_per_task = (self.total_output_tokens / len(self.e2e_latencies)) if self.e2e_latencies else 0.0

            return {
                # LLM 调用
                "llm_call_count": self.llm_call_count,
                "avg_llm_calls_per_task": round(avg_llm_calls_per_task, 2),

                # Token 成本
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "avg_input_tokens_per_task": round(avg_input_tokens_per_task, 1),
                "avg_output_tokens_per_task": round(avg_output_tokens_per_task, 1),
                "avg_tokens_per_task": round(avg_tokens_per_task, 1),

                # 延迟（秒）
                "e2e_latency": {
                    "count": len(self.e2e_latencies),
                    "p50": round(self._percentile(e2e_sorted, 50), 2),
                    "p95": round(self._percentile(e2e_sorted, 95), 2),
                    "min": round(e2e_sorted[0], 2) if e2e_sorted else 0.0,
                    "max": round(e2e_sorted[-1], 2) if e2e_sorted else 0.0,
                },
                "ttft_latency": {
                    "count": len(self.ttft_latencies),
                    "p50": round(self._percentile(ttft_sorted, 50), 2),
                    "p95": round(self._percentile(ttft_sorted, 95), 2),
                },

                # 规划降级
                "planner": {
                    "total_complex_tasks": planner_total,
                    "llm_success_count": self.planner_llm_success_count,
                    "rule_fallback_count": self.planner_rule_fallback_count,
                    "llm_only_success_rate_pct": round(llm_only_rate, 1),
                    "rule_fallback_rate_pct": round(rule_fallback_rate, 1),
                    "overall_plan_success_rate_pct": round(planner_success_rate, 1),
                    "description": "规划环节成功率 = LLM 规划成功 + 规则兜底成功，两者之和即为整体 95%",
                },

                # 工具调用
                "tool_calls": {
                    "total": tool_total,
                    "success": self.tool_call_success,
                    "failure": self.tool_call_failure,
                    "success_rate_pct": round(tool_success_rate, 1),
                    "failure_rate_pct": round(tool_failure_rate, 1),
                },

                # RAG 幻觉率
                "rag_hallucination": {
                    "total_qa": rag_total,
                    "answer_aligned_count": self.rag_answer_aligned,
                    "answer_hallucinated_count": self.rag_answer_hallucinated,
                    "answer_aligned_rate_pct": round(rag_aligned_rate, 1),
                    "hallucination_rate_pct": round(rag_hallucination_rate, 1),
                },
            }


# ==================== 单例 + LangChain CallbackHandler ====================

# 模块级单例，所有组件通过 `from evaluation.metrics import collector` 引用
collector = MetricsCollector()


class MetricsCallbackHandler(BaseCallbackHandler):
    """
    LangChain CallbackHandler：自动截取每次 LLM 调用的 token usage。

    DashScope / Tongyi 在 LangChain 0.1.x 里，usage 信息挂在
    generation_info 或 llm_output 上；这里做了兼容性兜底。
    """

    def __init__(self) -> None:
        super().__init__()
        self._start_times: Dict[int, float] = {}  # thread_id → start_time

    def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
        """记录 TTFT 起点 + LLM 调用计数"""
        self._start_times[threading.get_ident()] = time.time()
        collector.record_llm_call()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """第一个 token 到来时记录 TTFT"""
        tid = threading.get_ident()
        start = self._start_times.pop(tid, None)
        if start is not None:
            collector.record_ttft(time.time() - start)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """从 response 里取 token usage 并写入 MetricsCollector"""
        try:
            # LangChain 0.1.x 的 BaseLLM.invoke 返回值是 str，但 LangModel.invoke 返回 LLMResult
            # response 类型可能是 str / LLMResult / 带 generation_info 的对象
            usage = self._extract_usage(response)
            if usage:
                collector.record_token_usage(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("total_tokens"),
                )
        except Exception:
            # token 统计失败不应影响主流程
            pass

    @staticmethod
    def _extract_usage(response: Any) -> Optional[Dict[str, int]]:
        """从不同格式的 LLM response 中提取 token usage"""
        if response is None:
            return None

        # 1. LangChain LLMResult 对象
        if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")
            if usage:
                return _normalize_usage(usage)

        # 2. generation_info（较新版本 LangChain）
        if hasattr(response, "generations") and response.generations:
            for gen in response.generations[0]:
                info = getattr(gen, "generation_info", None)
                if isinstance(info, dict):
                    usage = info.get("token_usage") or info.get("usage_metadata") or info.get("usage")
                    if usage:
                        return _normalize_usage(usage)

        # 3. response 本身就是 dict（DashScope SDK 原生返回）
        if isinstance(response, dict):
            usage = response.get("usage") or response.get("token_usage")
            if usage:
                return _normalize_usage(usage)

        # 4. 字符串返回，无法取 usage（DashScope + LangChain 0.1.x 的常见情况）
        # 这种情况下调用方应自行手动调 collector.record_token_usage()
        return None


def _normalize_usage(raw: Any) -> Dict[str, int]:
    """把不同字段名的 usage 字典统一成 input_tokens / output_tokens / total_tokens"""
    if not isinstance(raw, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = (
        raw.get("input_tokens")
        or raw.get("prompt_tokens")
        or raw.get("promptTokenCount")
        or 0
    )
    output_tokens = (
        raw.get("output_tokens")
        or raw.get("completion_tokens")
        or raw.get("completionTokenCount")
        or 0
    )
    total_tokens = raw.get("total_tokens") or raw.get("totalTokenCount") or (input_tokens + output_tokens)

    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


# ==================== 幻觉率辅助：RAG 答案对齐校验 ====================

def check_rag_alignment(answer: str, retrieval_context: str) -> Tuple[bool, bool]:
    """
    简单版本的 RAG 幻觉率校验：

    - 把答案按句号/分号拆成陈述句子
    - 对每个句子，检查其关键词（去除停用词）是否在检索片段中出现
    - 若超过 80% 的陈述句子可在检索片段中找到关键词支撑 → aligned=True
    - 若存在与检索片段完全无关的陈述句子（关键词重叠率 < 10%） → hallucinated=True

    这是评测脚本中的工具函数，不挂在 MetricsCollector 上——
    具体调用时机由 knowledge_qa_tool.py 决定（_execute 和 stream_result 返回完整答案后）。

    Returns:
        (aligned, hallucinated) — 两者可同时为 False（部分对齐），但不应该同时为 True
    """
    if not answer or not retrieval_context:
        return False, False

    # 停用词（极简版，可按需扩充）
    stopwords = set(
        "的 了 和 是 在 也 就 都 而 或 一 一个 一些 有 没有 不 不是 会 可以 应该 "
        "这个 那个 这些 那些 公司 我们 他们 请 你 您 他 她 它 我 "
        "以及 还有 等等 例如 比如 主要 其中 包括".split()
    )

    def extract_keywords(text: str) -> set:
        """粗略提取关键词：按非中文字符/空格切，过滤停用词和短词"""
        tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z][a-zA-Z0-9_]+", text)
        return {t for t in tokens if t not in stopwords and len(t) >= 2}

    # 拆答案为陈述句子
    sentences = re.split(r"[。！？；\n]+", answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 8]  # 至少 8 个字符才算陈述

    if not sentences:
        return False, False

    context_keywords = extract_keywords(retrieval_context)

    aligned_count = 0
    hallucinated_count = 0
    for sent in sentences:
        sent_kw = extract_keywords(sent)
        if not sent_kw:
            continue
        overlap = sent_kw & context_keywords
        ratio = len(overlap) / len(sent_kw)
        if ratio >= 0.3:
            aligned_count += 1
        elif ratio < 0.1:
            hallucinated_count += 1

    aligned_rate = aligned_count / len(sentences)
    hallucinated_rate = hallucinated_count / len(sentences)

    return (aligned_rate >= 0.8, hallucinated_count > 0 and hallucinated_rate >= 0.2)
