"""
生产级 JSON 数组提取：O(N) 状态机全候选扫描 + 严格解析优先 + 降级修复兜底
零外部依赖（repair_json 仅作为可选兜底），兼顾容错与性能
"""
import json
import re
import logging
from dataclasses import dataclass
from typing import Any

try:
    from json_repair import repair_json
except Exception:
    repair_json = None

log = logging.getLogger("json_array_extractor")


@dataclass(frozen=True)
class ExtractResult:
    raw: str            # 从原文截取的原始片段
    fixed: str          # 可直接 json.loads 的字符串（严格命中时 == raw）
    data: list          # 解析后的 Python 对象
    repaired: bool      # 是否走了降级修复
    truncated: bool     # 原始片段是否未闭合（模型被截断）
    index: int          # 片段在原文中的起始偏移


_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s* $ ")


def strip_code_fence(text: str) -> str:
    """剥掉 ```json ... ``` 围栏。"""
    return _FENCE_RE.sub("", text)


def scan_bracket_spans(text: str) -> list[tuple[int, int, bool, int]]:
    """
    单次 O(N) 扫描，收集所有 [] 跨度。
    返回 (start, end, closed, depth)，按 start 升序；字符串内的括号不参与配对。
    """
    spans: list[tuple[int, int, bool, int]] = []
    stack: list[tuple[int, int]] = []
    in_string = False
    is_escape = False

    for i, ch in enumerate(text):
        if is_escape:                       # 转义字符整体跳过（仅在字符串内有意义）
            is_escape = False
            continue
        if in_string:
            if ch == "\\":
                is_escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "[":
            stack.append((i, len(stack)))
        elif ch == "]":
            if stack:
                start, depth = stack.pop()
                spans.append((start, i, True, depth))

    for start, depth in stack:              # 未闭合：模型输出被截断
        spans.append((start, len(text) - 1, False, depth))

    spans.sort(key=lambda s: s[0])
    return spans


def validate_payload(data: Any, expect_objects: bool = True, min_items: int = 1) -> bool:
    """业务校验：必须是（非空）对象数组，拦下 ["系统手册"] / ['A','B'] 这类伪命中。"""
    if not isinstance(data, list) or len(data) < min_items:
        return False
    if expect_objects:
        return all(isinstance(item, dict) for item in data)
    return True


def unwrap_dict(data: Any) -> list | None:
    """模型偶尔输出 {"plan": [...]}，把里面的对象数组捞出来。"""
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
                return value
    return None


def extract_json_array(
    text: str,
    expect_objects: bool = True,
    min_items: int = 1,
    max_repairs: int = 32,
    min_repair_ratio: float = 0.4,
    max_text_len: int = 200_000,
) -> ExtractResult | None:
    if not text or not text.strip():
        return None
    if len(text) > max_text_len:            # 防御超长输入
        text = text[:max_text_len]

    body = strip_code_fence(text)
    spans = scan_bracket_spans(body)
    if not spans:
        return None

    # 顶层候选优先；嵌套数组留给兜底，避免对同一个数组反复解析
    primary = [(s, e, c, body[s:e + 1]) for s, e, c, d in spans if d == 0 and e - s + 1 >= 2]
    nested = [(s, e, c, body[s:e + 1]) for s, e, c, d in spans if d > 0 and e - s + 1 >= 2]

    def resolve(parsed: Any):
        return parsed if validate_payload(parsed, expect_objects, min_items) else unwrap_dict(parsed)

    # ---- 第二级：严格解析（零改写，最快最可信）----
    for start, end, closed, seg in primary:
        try:
            parsed = json.loads(seg)
        except Exception:
            continue
        payload = resolve(parsed)
        if payload is not None:
            return ExtractResult(seg, seg, payload, False, not closed, start)

    # ---- 第三级：降级修复兜底（昂贵，限量 + sanity check）----
    if repair_json is not None:
        for start, end, closed, seg in (primary + nested)[:max_repairs]:
            try:
                repaired_str = repair_json(seg)
                if not isinstance(repaired_str, str):
                    continue
                # 修复不能把片段改得面目全非（防止把废话括号"修"成业务数据）
                if len(repaired_str) < max(2, int(len(seg) * min_repair_ratio)):
                    continue
                parsed = json.loads(repaired_str)
            except Exception:
                continue
            payload = resolve(parsed)
            if payload is not None:
                return ExtractResult(seg, repaired_str, payload, True, not closed, start)

    return None


def robust_parse_and_repair(text: str, expect_objects: bool = True) -> str | None:
    """兼容旧接口：成功返回 JSON 字符串，失败返回 None。"""
    result = extract_json_array(text, expect_objects=expect_objects)
    return result.fixed if result else None

if __name__ == "__main__":
    test_cases = [
        # 场景1：纯废话包围
        "好的，为您生成的数据如下：[{\"id\": 1, \"name\": \"A\"}]，希望对您有帮助！",
        
        # 场景2：单引号作祟 (需要 repair)
        "数据： [{'id': 1, 'name': 'A'}, {'id': 2, 'name': 'B'}]",
        
        # 场景3：缺逗号/缺引号等语法错误 (需要 repair)
        "提取结果：[{id: 1, name: 'A'}, {id: 2, name: 'B'}]",
        
        # 场景4：大模型输出截断/没闭合 (需要 repair)
        "数据如下：[{\"id\": 1, \"name\": \"A\"}, {\"id\": 2, \"na", 
        
        # 场景5：废话里包含方括号干扰，且JSON里有字符串方括号
        "请参考[系统手册]，结果：[{\"id\": 1, \"remark\": \"测试[通过]\"}]",
        
        # 场景6：大模型彻底摆烂，输出了非对象数组
        "最终结果：['A', 'B', 'C']" # 应该返回 None 或继续找下一个
    ]

    for idx, case in enumerate(test_cases):
        print(f"【场景{idx+1}】输入:\n{case}")
        result = robust_parse_and_repair(case)
        print(f"修复并解析后的结果:\n{result}")
        print("-" * 40)