#!/usr/bin/env python3
# 挂接层：工具分发——扩展模型能触达的能力。
"""
s02_tool_use.py - 工具

s01 里的智能体循环没有变，只是在工具数组里加了更多工具，
并用「名称 -> 处理函数」的分发表来路由调用。

    +----------+      +---------+      +------------------+
    |  用户    | ---> |  模型   | ---> |  工具分发表        |
    |  输入    |      |  (LLM)  |      |  各名 -> 各函数    |
    +----------+      +---------+      +------------------+
                          ^                 |
                          |  tool_result 回传
                          +-----------------+

要点：「循环本身一点没变，只是多了工具。」
"""

# 与 s01 相比：多工具时不必改循环，只增 TOOLS 与分发表。学习重点在 TOOL_HANDLERS。
# ---------------------------------------------------------------------------
import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()  # 一切文件工具相对此目录，避免把绝对路径当参数传来传去
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


def safe_path(p: str) -> Path:
    # 把相对路径钉在工作区内：resolve 后必须仍是 WORKDIR 子路径，防 .. 跳出沙箱
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- 分发表：模型在 tool_use 里给的是 `name` 与 `input`；用 name 找 Python 函数并 **展开 input 当 kwargs**
# dispatch map 将工具名映射到处理函数
# 这样新增工具 = 在 TOOLS 里声明 + 在 HANDLERS 里挂一行，agent_loop 不用改
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# 工具
# name：工具名称，description：工具描述
# 不同于s01，增加了处理函数。路径沙箱防止逃逸工作区。
# input_schema：规范了怎么传参，和强制约束与验证
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]


def agent_loop(messages: list):
    """与 s01 同结构；唯一区别是 for 里按 `block.name` 跳转到不同实现。"""
    while True:
        # 发消息 + 系统提示 + 工具模式定义
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        # 把助手整段内容（含文本块与 tool_use 块）记入历史
        messages.append({"role": "assistant", "content": response.content})
        # 本回合没有工具调用，结束本轮
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # `block.name` 必须出现在 TOOL_HANDLERS，否则应返回可读的 “Unknown tool”
                handler = TOOL_HANDLERS.get(block.name)
                # `block.input` 是 dict，与 tools 的 input_schema 一致；**kwargs 展开传入
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                print(f"> {block.name}:")
                print(output[:200])
                # 用 tool_result 对应该次 tool_use 的 id
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        # 工具结果以 user 角色发回，进入下一轮
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    # `history` 中顺序示例：
    # user(文本) -> assistant(多段，可能含多 tool_use) -> user(多 tool_result) -> assistant(…) …
    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
