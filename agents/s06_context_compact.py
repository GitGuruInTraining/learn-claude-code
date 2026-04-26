#!/usr/bin/env python3
# 挂接层：压缩——清内存，让超长会话能一直跑下去。
"""
s06_context_compact.py - 上下文压缩

三层压缩管道，让智能体理论上可以「永远」工作下去：

    每一轮：
    +------------------+
    | 工具调用结果      |
    +------------------+
            |
            v
    [第 1 层：micro_compact]        （每轮静默执行）
      将最近 3 条以外的、非 read_file 的 tool_result
      替换为一句短占位，标明曾调用的工具名（具体格式见本文件 `micro_compact`）
            |
            v
    [判断：token 是否 > 50000？]
       |               |
      否              是
       |               |
       v               v
    继续         [第 2 层：auto_compact]
                  把完整转写存到 .transcripts/
                  让 LLM 总结对话
                  用 [摘要] 替换全部 messages
                        |
                        v
                [第 3 层：compact 工具]
                  模型主动调用 compact -> 立即走与上类似的总结
                  与自动压缩逻辑相同，只是手动触发

要点：「有策略地遗忘，才能无限期接着干。」
"""

# 学习重点：① 轻量去旧（micro） ② 超阈值时整单摘要（auto） ③ 给模型一个「自己喊停并压历史」的 compact 工具
# `messages` 为 list，多处原地修改 `[:]` 即替换同一引用，调用方也可见。
# ---------------------------------------------------------------------------
import json
import os
import subprocess
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

# 与真实 tokenizer 有偏差，仅作「需不需要 auto_compact」的粗开关
THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"  # 全量落盘，便于出事后从 jsonl 找回
# 只保留**最近**这么多个 tool_result 的原文，更早的用占位条替换
KEEP_RECENT = 3
# 不压缩 read_file 的长结果：多为代码片段，压掉会丢细节并逼模型重读
PRESERVE_RESULT_TOOLS = {"read_file"}


def estimate_tokens(messages: list) -> int:
    """粗算 token 数：约 4 个字符折合 1 token。"""
    return len(str(messages)) // 4


# -- 第 1 层：micro_compact，用占位符压缩陈旧工具结果 --
def micro_compact(messages: list) -> list:
    # 在 `user` 且 `content` 为**列表**的那几条里找 tool_result（与 API 的块结构一致）
    # 收集所有 tool_result 的 (消息下标, 片段下标, 字典) 元组
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    # 根据先前 assistant 消息里的 tool_use_id 反查每个结果对应的工具名
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name
    # 清空较旧结果（保留最近 KEEP_RECENT 条）。read_file 的结果保留，因为
    # 多作为参考资料；压掉会迫使智能体重新读盘。
    to_clear = tool_results[:-KEEP_RECENT]
    for _, _, result in to_clear:
        if not isinstance(result.get("content"), str) or len(result["content"]) <= 100:
            continue
        tool_id = result.get("tool_use_id", "")
        tool_name = tool_name_map.get(tool_id, "unknown")
        if tool_name in PRESERVE_RESULT_TOOLS:
            continue
        result["content"] = f"[Previous: used {tool_name}]"
    return messages


# -- 第 2 层：auto_compact，落盘、总结、用单条摘要替换全历史 --
def auto_compact(messages: list) -> list:
    # 完整转写落盘
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    # 只取**尾部** 8w 字符避免摘要请求体过长；`default=str` 用于序列化非 JSON 原生的块对象
    # 请求 LLM 生成摘要
    conversation_text = json.dumps(messages, default=str)[-80000:]
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}],
        max_tokens=2000,
    )
    summary = next((block.text for block in response.content if hasattr(block, "text")), "")
    if not summary:
        summary = "No summary generated."
    # 用压缩后的单条 user 消息替换整段历史
    return [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
    ]


# -- 工具实现（与前面章节同构）--
def safe_path(p: str) -> Path:
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
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
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


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # 实际压缩不在 handler 里做，而靠 agent_loop 看到 compact 后调 auto_compact
    "compact":    lambda **kw: "Manual compression requested.",
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "compact", "description": "Trigger manual conversation compression.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}},
]


def agent_loop(messages: list):
    """
    每轮：先瘦身 messages -> 再（可选）整单摘要
    然后正常走 tools；若本回合有 compact 工具，则在写入 tool_result 后**立刻**全量压并 return，
    这样 REPL 会在下一轮用「短历史」再说话。
    """
    while True:
        # 第 1 层：每次调 LLM 前先做 micro_compact
        micro_compact(messages)
        # 第 2 层：估计 token 超阈值则自动全量压缩
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            # `[:]` 原地替换，保证主函数里的 history 也指向新短列表
            messages[:] = auto_compact(messages)
        # 在微压缩/可能自动全量压之后的上下文中再调 LLM
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        manual_compact = False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact":
                    # 仅占位，真正压缩在循环末尾
                    manual_compact = True
                    output = "Compressing..."
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    try:
                        output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                    except Exception as e:
                        output = f"Error: {e}"
                print(f"> {block.name}:")
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        messages.append({"role": "user", "content": results})
        # 第 3 层：由 compact 工具触发的手动全量压缩
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
            return


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
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
