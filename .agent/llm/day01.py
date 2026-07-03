# -*- coding: utf-8 -*-
import os
import sys
import json
import ast
import operator
from openai import OpenAI
last_response = None

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    # api_key="sk-ad756abc376f48b9b1d1c220c948898f",
    base_url="https://api.deepseek.com",  # OpenAI/Qwen/DeepSeek/vLLM 主要换这里
)
# 2. messages 是整个对话历史
# system 一般只放一次，用来定义 AI 的身份、风格、规则
messages = [
    {
        "role": "system",
        "content": """
                    你是一个专业的AI工程助教，回答要清晰、循序渐进、适合初学者。你现在可以使用 calculator 工具进行算术计算。
                    规则：
                    1. 只要用户的问题涉及明确算术计算，就优先调用 calculator 工具。
                    2. 不要自己心算复杂数字。
                    3. 工具返回结果后，再用自然语言解释给用户。"""
    }
]
# =========================
# 1. 安全计算器工具
# =========================

ALLOWED_OPERATORS = {
    ast.Add: operator.add,        # +
    ast.Sub: operator.sub,        # -
    ast.Mult: operator.mul,       # *
    ast.Div: operator.truediv,    # /
    ast.FloorDiv: operator.floordiv,  # //
    ast.Mod: operator.mod,        # %
    ast.Pow: operator.pow,        # **
    ast.USub: operator.neg,       # 负数，比如 -3
    ast.UAdd: operator.pos,       # 正数，比如 +3
}

def safe_eval(node):
    """
    只允许计算数字和基础算术表达式。
    不允许执行函数、变量、文件、系统命令。
    """

    if isinstance(node, ast.Expression):
        return safe_eval(node.body)

    # Python 3.8+
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("只支持数字计算")

    # 兼容旧版本 Python
    if isinstance(node, ast.Num):
        return node.n

    # 二元运算：1 + 2、3 * 4、2 ** 3
    if isinstance(node, ast.BinOp):
        left = safe_eval(node.left)
        right = safe_eval(node.right)

        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"不支持的运算符：{op_type}")

        return ALLOWED_OPERATORS[op_type](left, right)

    # 一元运算：-3、+5
    if isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)

        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"不支持的一元运算符：{op_type}")

        return ALLOWED_OPERATORS[op_type](operand)

    raise ValueError("表达式里包含不允许的内容")


def calculator(expression: str) -> str:
    """
    calculator 工具：
    输入一个算术表达式字符串，返回计算结果。
    """

    try:
        tree = ast.parse(expression, mode="eval")
        result = safe_eval(tree)
        return str(result)

    except Exception as e:
        return f"计算失败：{e}"


# =========================
# 2. 定义 tools schema
# =========================

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "用于计算基础算术表达式，支持 +、-、*、/、//、%、** 和括号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "需要计算的算术表达式，例如：123 * 456 + 789"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]
def chat(user_input: str) -> str:
    """
    带工具调用能力的多轮对话：
    1. 追加 user message
    2. 调模型
    3. 如果模型要调用工具，就执行工具
    4. 把 tool 结果追加进 messages
    5. 再调模型，让模型生成最终回答
    """

    global last_response

    messages.append({
        "role": "user",
        "content": user_input
    })

    # 最多允许连续工具调用 5 轮，防止死循环
    for _ in range(5):
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=1024,
        )

        last_response = response

        assistant_message = response.choices[0].message

        # 把 assistant message 转成 dict，追加进 messages
        assistant_message_dict = assistant_message.model_dump(exclude_none=True)
        messages.append(assistant_message_dict)

        # 如果没有 tool_calls，说明模型已经给出最终回答
        if not assistant_message.tool_calls:
            return assistant_message.content

        # 如果有 tool_calls，执行每一个工具
        for tool_call in assistant_message.tool_calls:
            tool_result = execute_tool_call(tool_call)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

    return "工具调用轮数过多，已停止。"


def execute_tool_call(tool_call):
    """
    根据模型请求的 tool_call，执行对应 Python 函数。
    """

    function_name = tool_call.function.name
    arguments_text = tool_call.function.arguments

    try:
        arguments = json.loads(arguments_text)
    except Exception as e:
        return f"工具参数 JSON 解析失败：{e}"

    if function_name == "calculator":
        expression = arguments.get("expression", "")
        return calculator(expression)

    return f"未知工具：{function_name}"
def print_history():
    print("\n========== 当前 messages 历史 ==========")
    for i, msg in enumerate(messages):
        print(f"\n[{i}] role = {msg['role']}")
        print(msg["content"])
    print("\n=======================================")
# 7. 主循环
print("多轮对话程序已启动。")
print("输入 exit / quit / 退出 可以结束。")
print("输入 history 可以查看当前 messages 历史。")
print("输入 clear 可以清空对话历史，但保留 system。")

while True:
    user_input = input("\n你：").strip()

    if user_input.lower() in ["exit", "quit", "退出"]:
        print("程序结束。")
        break

    if user_input.lower() == "history":
        print_history()
        continue

    if user_input.lower() == "clear":
        messages.clear()
        messages.append({
            "role": "system",
            "content": "你是一个专业的AI工程助教，回答要清晰、循序渐进、适合初学者。"
        })
        print("已清空历史，只保留 system 消息。")
        continue

    if not user_input:
        print("请输入内容。")
        continue

    try:
        answer = chat(user_input)
        print("\nAI：", answer)

    except Exception as e:
        print("\n调用模型出错：", e)