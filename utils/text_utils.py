"""
utils/text_utils.py — 纯文本处理工具

从 workflow_base.py 拆出的无状态工具函数，用于 JSON/Python 代码提取和文本格式化。
所有函数均为静态方法或模块级函数，不依赖任何工作流状态。
"""
import re
import ast
import json
from typing import List, Dict, Optional


def extract_json(text: str) -> str:
    """
    从 LLM 响应文本中提取 JSON 字符串。
    按优先级依次尝试：```json 块 → ``` 块 → 直接解析 → 正则提取 → 原文。
    """
    # 1. ```json ... ```
    matches = re.findall(r'```json\s*\n(.*?)\n?\s*```', text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # 2. ``` ... ``` (generic code block containing JSON)
    matches = re.findall(r'```\s*\n(.*?)\n?\s*```', text, re.DOTALL)
    for m in sorted(matches, key=len, reverse=True):
        if m.strip().startswith('{') and m.strip().endswith('}'):
            return m.strip()

    # 3. Entire text is valid JSON
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass

    # 4. Regex extract largest {...} block
    candidates = re.findall(r'(\{[\s\S]*\})', text)
    if candidates:
        longest = max(candidates, key=len)
        try:
            json.loads(longest)
            return longest
        except (json.JSONDecodeError, ValueError):
            pass

    return text.strip()


def extract_python(text: str) -> str:
    """
    从 LLM 响应文本中提取 Python 代码。
    按优先级：```python 块 → ``` 块 → 识别 import/class/def 起始行。
    """
    code = text.strip()

    # 1. ```python ... ```
    matches = re.findall(r'```python\s*\n(.*?)\n?\s*```', text, re.DOTALL)
    if matches:
        code = max(matches, key=len).strip()
    else:
        # 2. ``` ... ```
        matches = re.findall(r'```\s*\n(.*?)\n?\s*```', text, re.DOTALL)
        if matches:
            code = max(matches, key=len).strip()
        else:
            # 3. Find first import/class/def line
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ', 'class ', 'def ', '@')):
                    code = '\n'.join(lines[i:]).strip()
                    break

    # Handle string-wrapped code
    if len(code) > 2 and (code.startswith('"') or code.startswith("'")):
        try:
            unescaped = ast.literal_eval(code)
            if isinstance(unescaped, str):
                return unescaped.strip()
        except (ValueError, SyntaxError):
            pass

    return code


def highlight_target_in_code(code: str, target: str) -> str:
    """
    在代码中高亮标记目标函数/区域，辅助 Coder agent 定位修改位置。
    """
    if target in ("imports", "main_block"):
        marker = {
            "imports": "# >>> TARGET: IMPORTS <<<",
            "main_block": "# >>> TARGET: MAIN BLOCK <<<"
        }.get(target, f"# >>> TARGET: {target} <<<")
        return f"{marker}\n{code}"

    lines = code.split('\n')
    highlighted = []
    in_target = False
    target_found = False

    for line in lines:
        if f"def {target}(" in line or f"async def {target}(" in line:
            highlighted.append(f"\n# >>> TARGET FUNCTION: {target} <<<")
            in_target = True
            target_found = True
        elif not target_found and "def " in line and target in line:
            highlighted.append(f"\n# >>> TARGET FUNCTION (Probable): {target} <<<")
            in_target = True
            target_found = True
        elif in_target and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            in_target = False
        highlighted.append(line)

    if not target_found:
        return f"# >>> TARGET: {target} (Not found in code, please implement) <<<\n{code}"
    return '\n'.join(highlighted)


def format_failure_histories(histories: List[Dict], max_entries: int = 3) -> str:
    """
    将失败历史格式化为人类可读的文本块。

    Args:
        histories: 失败历史记录列表。
        max_entries: 最多展示的条目数。

    Returns:
        格式化的失败历史文本，空列表时返回空字符串。
    """
    if not histories:
        return ""

    formatted = "\n### ⚠️ PAST FAILURES (Avoid Repeating These Errors) ###\n"
    for i, hist in enumerate(histories[-max_entries:], 1):
        iter_num = hist.get('iteration', '?')
        timestamp = hist.get('timestamp', '').split()[1] if hist.get('timestamp') else ''
        error_type = hist.get('ticket_assigned_to', 'Unknown')
        fix_target = hist.get('fix_target', 'N/A')
        analysis = hist.get('analysis', 'N/A').replace('\n', ' ').strip()
        evidence = hist.get('evidence', '').replace('\n', ' ').strip()[:100]
        formatted += (
            f"\n[Iter {iter_num} | {timestamp}] {error_type} → {fix_target}\n"
            f"  Cause: {analysis[:200]}...\n"
        )
        if evidence:
            formatted += f"  Evidence: {evidence}...\n"
    return formatted
