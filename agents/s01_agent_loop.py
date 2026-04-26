#!/usr/bin/env python3
# 挂接层：核心循环——模型与真实世界的第一次连接。
"""
s01_agent_loop.py - 智能体循环

AI 编程智能体的全部秘诀就藏在一个模式里：

    当 停止原因 仍为 "tool_use" 时循环：
        response = 调模型(消息, 工具定义)
        执行工具
        把结果追加进对话

    +----------+      +---------+      +---------+
    |  用户    | ---> |  模型   | ---> |  工具   |
    |  输入    |      |  (LLM)  |      |  执行   |
    +----------+      +---------+      +---------+
                          ^                 |
                          |  tool_result 回传
                          +-----------------+
                          （如此循环，直到不再请求工具）

核心循环：把工具结果不断喂回模型，直到模型决定停止。
生产级智能体还会在此之上叠加策略、钩子与生命周期控制。
"""

# ---------------------------------------------------------------------------
# 环境：.env 里放 MODEL_ID，可选 ANTHROPIC_BASE_URL 指向兼容网关
# 学习重点：本文件只演示「发请求 → 执行工具 → 把结果当 user 发回去」的闭环。
# ---------------------------------------------------------------------------
import os
import subprocess

try:
    import readline
    # 交互式 REPL 在 macOS 上若中文/退格异常，可尝试这些 readline 绑定（#143）
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)  # 将 .env 写入 os.environ，供下面客户端读 MODEL

# 若使用自定义网关（ANTHROPIC_BASE_URL），避免与官方 SDK 的 ANTHROPIC_AUTH_TOKEN 冲突
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# Anthropic 客户端；base_url 为空时使用官方默认端点
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# 系统提示：教模型「你是谁、在哪工作、怎么做事」；不在这里列 bash 语法，而在 tools 里声明能力
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# 暴露给模型的唯一工具：bash
# 注意：name / description / input_schema 会原样进 API，模型依此决定何时、如何带参数调用
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


def run_bash(command: str) -> str:
    """本地执行模型给出的 shell 串：教学用弱防护 + 限长/超时。返回字符串会进入 tool_result。"""
    # 极粗糙黑名单：真产品里要用更完整的策略/沙箱
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # shell=True 方便一行命令，但有注入风险，生产需谨慎
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        # 截断避免 tool_result 撑爆下文 token；与模型可读的 8k/窗口无关，只是防护
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# -- 核心模式：在模型停止前不断用 while 循环调用工具 --
def agent_loop(messages: list):
    """
    每一轮：请求模型 -> 把 assistant 消息追加进 history
    若 `stop_reason == "tool_use"`，必须按块执行工具并把结果写回，否则无法完成协议。
    tool_use_id 与 tool_result 一一对应，见 Anthropic 消息格式说明。
    """
    while True:
        # 1) 发完整对话 + system + 工具模式；max_tokens 限制**本轮**模型输出长度
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        # 2) 无论是否有工具，先把 assistant 的整段 `content`（可能含多段 text + tool_use）存进历史
        messages.append({"role": "assistant", "content": response.content})
        # 3) 模型以纯文本收束、不再要工具时退出循环（本示例里常是「说完了」）
        if response.stop_reason != "tool_use":
            return
        # 4) 否则必须处理**本次**所有 tool_use 块，拼成**一条** user 消息（content 为 list）
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # block.id / block.input 由 API 反序列化好；本例只有 name==bash
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                print(output[:200])
                # type 固定为 tool_result；必须带对应该 tool_use 的 id，否则下一轮模型对不上
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})
        # 5) 工具结果在协议里通常作为 user 发回（不是 assistant），这样下一轮模型才能继续想
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    # 整个会话共用一个 `history` 列表，多轮用户输入会不断 append，agent_loop 可能追加多轮
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        # 新的人类输入以一条 user 消息进入（content 是字符串，不是 list）
        history.append({"role": "user", "content": query})
        # 跑循环直到本轮模型不再要工具；history 会原地被追加 assistant / user(tool_result) 多段
        agent_loop(history)
        # 循环结束后，最后一条应已是 assistant，且以文本块收束
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
