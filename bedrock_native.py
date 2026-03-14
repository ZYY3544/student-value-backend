"""
===========================================
Bedrock 原生 SDK 调用模块
===========================================
所有 Bedrock 调用统一使用 boto3 原生 SDK（invoke_model / invoke_model_with_response_stream）。
不使用 BedrockOpenAIClient 或任何 OpenAI 兼容封装。

用法：
    from bedrock_native import create_bedrock_client, bedrock_invoke, bedrock_stream_text

    client = create_bedrock_client()
    result = bedrock_invoke(client, model, messages, temperature=0.5)
    print(result["content"])
"""

import json
import boto3
from config import config


def create_bedrock_client():
    """创建共享的 boto3 bedrock-runtime 客户端"""
    return boto3.client(
        'bedrock-runtime',
        region_name=config.AWS_REGION,
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    )


# ===========================================
# 消息格式转换（OpenAI → Anthropic Messages API）
# ===========================================

def _extract_system_and_messages(messages):
    """从 OpenAI 格式消息中分离 system prompt，转换为 Anthropic Messages API 格式"""
    system_text = ""
    anthropic_messages = []

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            msg_tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id", "")
        else:
            role = getattr(msg, 'role', 'user')
            content = getattr(msg, 'content', '')
            msg_tool_calls = getattr(msg, 'tool_calls', None)
            tool_call_id = getattr(msg, 'tool_call_id', '')

        if role == "system":
            system_text = content

        elif role == "assistant":
            if msg_tool_calls:
                content_blocks = []
                if content:
                    content_blocks.append({"type": "text", "text": content})
                for tc in msg_tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get('function', {})
                        tc_id = tc.get('id', '')
                        fn_name = fn.get('name', '')
                        fn_args = fn.get('arguments', '{}')
                    else:
                        fn = tc.function
                        tc_id = tc.id
                        fn_name = fn.name
                        fn_args = fn.arguments
                    try:
                        input_data = json.loads(fn_args) if isinstance(fn_args, str) else fn_args
                    except json.JSONDecodeError:
                        input_data = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc_id,
                        "name": fn_name,
                        "input": input_data
                    })
                anthropic_messages.append({"role": "assistant", "content": content_blocks})
            else:
                # Anthropic API 不接受空字符串 content，必须有实际文本
                safe_content = content if content else "好的。"
                anthropic_messages.append({"role": "assistant", "content": safe_content})

        elif role == "tool":
            tool_content = content if isinstance(content, str) else json.dumps(content)
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": tool_content
            }
            if (anthropic_messages
                    and anthropic_messages[-1]["role"] == "user"
                    and isinstance(anthropic_messages[-1]["content"], list)
                    and anthropic_messages[-1]["content"]
                    and anthropic_messages[-1]["content"][0].get("type") == "tool_result"):
                anthropic_messages[-1]["content"].append(tool_result_block)
            else:
                anthropic_messages.append({
                    "role": "user",
                    "content": [tool_result_block]
                })

        else:  # user
            anthropic_messages.append({"role": "user", "content": content or ""})

    return system_text, anthropic_messages


def _convert_tools(tools):
    """将 OpenAI tools 格式转换为 Anthropic tools 格式"""
    if not tools:
        return []
    anthropic_tools = []
    for tool in tools:
        fn = tool.get("function", {})
        anthropic_tools.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}})
        })
    return anthropic_tools


def _build_body(messages, system=None, temperature=0.7, max_tokens=4096, tools=None, response_format=None):
    """构建 Anthropic Messages API 请求体"""
    system_text, anthropic_messages = _extract_system_and_messages(messages)
    if system:
        system_text = system

    if response_format and response_format.get("type") == "json_object":
        system_text = (system_text or "") + "\n\n[CRITICAL] You MUST respond with valid JSON only. No markdown, no explanation, no extra text — just a single JSON object."

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": anthropic_messages,
    }
    if system_text:
        body["system"] = system_text

    anthropic_tools = _convert_tools(tools)
    if anthropic_tools:
        body["tools"] = anthropic_tools

    return body


# ===========================================
# 同步调用
# ===========================================

def bedrock_invoke(client, model, messages, system=None, temperature=0.7,
                   max_tokens=4096, tools=None, response_format=None, **_kwargs):
    """
    同步调用 Bedrock invoke_model

    Returns:
        dict: {
            "content": str,
            "tool_calls": list[dict] | None,  # each: {id, name, arguments(str)}
            "finish_reason": "stop" | "tool_use",
            "usage": {input_tokens, output_tokens}
        }
    """
    body = _build_body(messages, system, temperature, max_tokens, tools, response_format)

    response = client.invoke_model(
        modelId=model,
        contentType='application/json',
        body=json.dumps(body)
    )
    result = json.loads(response['body'].read())

    text_content = ""
    tool_calls = []
    for block in result.get("content", []):
        if block.get("type") == "text":
            text_content += block.get("text", "")
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": json.dumps(block.get("input", {}))
            })

    return {
        "content": text_content,
        "tool_calls": tool_calls if tool_calls else None,
        "finish_reason": "tool_use" if tool_calls else "stop",
        "usage": result.get("usage", {}),
    }


# ===========================================
# 流式调用
# ===========================================

def bedrock_stream(client, model, messages, system=None, temperature=0.7,
                   max_tokens=4096, tools=None, **_kwargs):
    """
    流式调用 Bedrock invoke_model_with_response_stream

    Yields:
        dict 事件:
        - {"type": "text", "text": "..."}
        - {"type": "tool_start", "index": int, "id": str, "name": str}
        - {"type": "tool_delta", "index": int, "partial_json": str}
        - {"type": "done", "finish_reason": "stop"|"tool_use", "usage": dict}
    """
    body = _build_body(messages, system, temperature, max_tokens, tools)

    response = client.invoke_model_with_response_stream(
        modelId=model,
        contentType='application/json',
        body=json.dumps(body)
    )

    tool_index = -1
    for event in response['body']:
        # 处理 Bedrock 流式错误事件
        if "internalServerException" in event:
            err = event["internalServerException"]
            raise RuntimeError(f"Bedrock 内部错误: {err.get('message', str(err))}")
        if "modelStreamErrorException" in event:
            err = event["modelStreamErrorException"]
            raise RuntimeError(f"Bedrock 流式错误: {err.get('message', str(err))}")
        if "throttlingException" in event:
            err = event["throttlingException"]
            raise RuntimeError(f"Bedrock 限流(429): {err.get('message', str(err))}")
        if "validationException" in event:
            err = event["validationException"]
            raise RuntimeError(f"Bedrock 参数校验失败: {err.get('message', str(err))}")

        chunk_bytes = event.get("chunk", {}).get("bytes", b"")
        if not chunk_bytes:
            continue

        data = json.loads(chunk_bytes)
        event_type = data.get("type", "")

        if event_type == "content_block_start":
            block = data.get("content_block", {})
            if block.get("type") == "tool_use":
                tool_index += 1
                yield {
                    "type": "tool_start",
                    "index": tool_index,
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                }

        elif event_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "text_delta":
                yield {"type": "text", "text": delta.get("text", "")}
            elif delta_type == "input_json_delta":
                partial = delta.get("partial_json", "")
                if partial:
                    yield {"type": "tool_delta", "index": tool_index, "partial_json": partial}

        elif event_type == "message_delta":
            stop_reason = data.get("delta", {}).get("stop_reason", "end_turn")
            finish = "tool_use" if stop_reason == "tool_use" else "stop"
            usage = data.get("usage", {})
            yield {"type": "done", "finish_reason": finish, "usage": usage}


def bedrock_stream_text(client, model, messages, system=None, temperature=0.7, max_tokens=4096):
    """
    简化的流式调用，只 yield 文本字符串（无工具支持）
    与 _stream_report_analysis 相同的模式
    """
    for event in bedrock_stream(client, model, messages, system, temperature, max_tokens):
        if event["type"] == "text":
            yield event["text"]


# ===========================================
# 统一调用接口（自动分发 Bedrock / GLM）
# ===========================================

def unified_invoke(client, model, provider, messages, **kwargs):
    """
    统一同步调用：Bedrock 走 boto3 原生，GLM 走 OpenAI SDK

    Returns:
        dict: {content, tool_calls, finish_reason, usage}
    """
    if provider in ("haiku", "sonnet"):
        return bedrock_invoke(client, model, messages, **kwargs)

    # GLM / OpenAI 兼容路径
    params = {"model": model, "messages": messages}
    if "temperature" in kwargs:
        params["temperature"] = kwargs["temperature"]
    if "max_tokens" in kwargs:
        params["max_tokens"] = kwargs["max_tokens"]
    if kwargs.get("response_format"):
        params["response_format"] = kwargs["response_format"]
    if kwargs.get("tools"):
        params["tools"] = kwargs["tools"]

    response = client.chat.completions.create(**params)
    choice = response.choices[0]

    tool_calls = None
    if choice.message.tool_calls:
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in choice.message.tool_calls
        ]

    return {
        "content": choice.message.content or "",
        "tool_calls": tool_calls,
        "finish_reason": choice.finish_reason or "stop",
        "usage": {
            "input_tokens": getattr(response.usage, 'prompt_tokens', 0),
            "output_tokens": getattr(response.usage, 'completion_tokens', 0),
        } if response.usage else {},
    }


def unified_stream(client, model, provider, messages, **kwargs):
    """
    统一流式调用：Bedrock 走 boto3 原生，GLM 走 OpenAI SDK

    Yields:
        dict 事件（同 bedrock_stream 格式）
    """
    if provider in ("haiku", "sonnet"):
        yield from bedrock_stream(client, model, messages, **kwargs)
        return

    # GLM / OpenAI 兼容路径
    params = {"model": model, "messages": messages, "stream": True}
    if "temperature" in kwargs:
        params["temperature"] = kwargs["temperature"]
    if kwargs.get("tools"):
        params["tools"] = kwargs["tools"]

    stream = client.chat.completions.create(**params)
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        finish = chunk.choices[0].finish_reason

        if delta.content:
            yield {"type": "text", "text": delta.content}

        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            for tc_delta in delta.tool_calls:
                if tc_delta.id:
                    yield {
                        "type": "tool_start",
                        "index": tc_delta.index,
                        "id": tc_delta.id,
                        "name": tc_delta.function.name if tc_delta.function else "",
                    }
                if tc_delta.function and tc_delta.function.arguments:
                    yield {
                        "type": "tool_delta",
                        "index": tc_delta.index,
                        "partial_json": tc_delta.function.arguments,
                    }

        if finish:
            yield {
                "type": "done",
                "finish_reason": "tool_use" if finish == "tool_calls" else "stop",
            }


def unified_stream_text(client, model, provider, messages, **kwargs):
    """统一文本流式调用，只 yield 文本字符串"""
    for event in unified_stream(client, model, provider, messages, **kwargs):
        if event["type"] == "text":
            yield event["text"]
