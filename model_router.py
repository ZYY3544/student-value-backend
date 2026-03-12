"""
===========================================
模型路由器 (Model Router)
===========================================
根据用户用量自动切换 Sonnet → GLM

核心逻辑：
1. 每个用户有 Sonnet 预算上限（默认 ¥15）
2. 未超预算 → 使用 AWS Bedrock Sonnet
3. 超出预算 → 自动降级到 GLM
4. 用量记录在 PostgreSQL

包含：
- UsageTracker: 用量追踪（PostgreSQL）
- BedrockOpenAIClient: AWS Bedrock 的 OpenAI 兼容适配器
- ModelRouter: 模型选择路由器
"""

import os
import json
import threading
from typing import Optional
from dataclasses import dataclass, field
from config import config


# ===========================================
# 用量追踪器 (PostgreSQL)
# ===========================================

class UsageTracker:
    """基于 PostgreSQL 的用户 LLM 用量追踪"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._init_table()

    def _init_table(self):
        """创建用量表（幂等）"""
        if not self.database_url:
            print("[UsageTracker] 未配置 DATABASE_URL，用量将不会持久化")
            return
        try:
            import psycopg2
            conn = psycopg2.connect(self.database_url)
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS user_llm_usage (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(200) NOT NULL,
                    provider VARCHAR(20) NOT NULL,
                    model VARCHAR(100) NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_rmb NUMERIC(10,6) NOT NULL DEFAULT 0,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            ''')
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_llm_usage_user_id
                ON user_llm_usage(user_id)
            ''')
            conn.commit()
            cur.close()
            conn.close()
            print("[UsageTracker] 用量表初始化成功")
        except Exception as e:
            print(f"[UsageTracker] 用量表初始化失败: {e}")

    def get_user_sonnet_cost(self, user_id: str) -> float:
        """获取用户已消耗的 Sonnet 费用（人民币）"""
        if not self.database_url or not user_id:
            return 0.0
        try:
            import psycopg2
            conn = psycopg2.connect(self.database_url)
            cur = conn.cursor()
            cur.execute(
                "SELECT COALESCE(SUM(cost_rmb), 0) FROM user_llm_usage WHERE user_id = %s AND provider = 'sonnet'",
                (user_id,)
            )
            cost = float(cur.fetchone()[0])
            cur.close()
            conn.close()
            return cost
        except Exception as e:
            print(f"[UsageTracker] 查询用量失败: {e}")
            return 0.0

    def record_usage(self, user_id: str, provider: str, model: str,
                     input_tokens: int, output_tokens: int, cost_rmb: float):
        """记录一次 LLM 调用的用量"""
        if not self.database_url:
            return
        try:
            import psycopg2
            conn = psycopg2.connect(self.database_url)
            cur = conn.cursor()
            cur.execute(
                '''INSERT INTO user_llm_usage (user_id, provider, model, input_tokens, output_tokens, cost_rmb)
                   VALUES (%s, %s, %s, %s, %s, %s)''',
                (user_id or 'anonymous', provider, model, input_tokens, output_tokens, cost_rmb)
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[UsageTracker] 记录用量失败: {e}")

    def get_user_usage_summary(self, user_id: str) -> dict:
        """获取用户用量摘要"""
        sonnet_cost = self.get_user_sonnet_cost(user_id)
        budget = config.SONNET_BUDGET_PER_USER
        return {
            "user_id": user_id,
            "sonnet_cost_rmb": round(sonnet_cost, 4),
            "sonnet_budget_rmb": budget,
            "sonnet_remaining_rmb": round(max(0, budget - sonnet_cost), 4),
            "budget_exceeded": sonnet_cost >= budget,
        }


# ===========================================
# Bedrock OpenAI 兼容适配器
# ===========================================
# 将 AWS Bedrock Converse API 包装为 OpenAI SDK 兼容接口
# 这样 multi_agent.py 中的所有代码无需修改

@dataclass
class _Choice:
    message: object = None
    delta: object = None
    finish_reason: str = None
    index: int = 0

@dataclass
class _Message:
    role: str = "assistant"
    content: str = ""
    tool_calls: list = None

@dataclass
class _Delta:
    content: str = None
    role: str = None
    tool_calls: list = None

@dataclass
class _ToolCall:
    id: str = ""
    type: str = "function"
    function: object = None

@dataclass
class _Function:
    name: str = ""
    arguments: str = ""

@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class _CompletionResponse:
    choices: list = field(default_factory=list)
    usage: _Usage = None


class _StreamChunk:
    def __init__(self, choices=None):
        self.choices = choices or []


class BedrockChatCompletions:
    """模拟 openai.chat.completions 接口，内部调用 AWS Bedrock Converse API"""

    def __init__(self, bedrock_client, default_model: str):
        self._client = bedrock_client
        self._default_model = default_model

    def _convert_messages(self, messages: list) -> tuple:
        """将 OpenAI 格式消息转换为 Bedrock 格式，分离 system prompt"""
        system_prompts = []
        bedrock_messages = []

        for msg in messages:
            # 兼容 dict 和 object 两种格式
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
                system_prompts.append({"text": content})

            elif role == "assistant":
                if msg_tool_calls:
                    content_blocks = []
                    if content:
                        content_blocks.append({"text": content})
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
                            "toolUse": {
                                "toolUseId": tc_id,
                                "name": fn_name,
                                "input": input_data
                            }
                        })
                    bedrock_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    bedrock_messages.append({"role": "assistant", "content": [{"text": content or ""}]})

            elif role == "tool":
                tool_content = content if isinstance(content, str) else json.dumps(content)
                bedrock_messages.append({
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": tool_call_id,
                            "content": [{"text": tool_content}]
                        }
                    }]
                })

            else:  # user
                bedrock_messages.append({"role": "user", "content": [{"text": content or ""}]})

        return system_prompts, bedrock_messages

    def _convert_tools(self, tools: list) -> dict:
        """将 OpenAI tools 格式转换为 Bedrock toolConfig"""
        if not tools:
            return {}

        bedrock_tools = []
        for tool in tools:
            fn = tool.get("function", {})
            bedrock_tools.append({
                "toolSpec": {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "inputSchema": {
                        "json": fn.get("parameters", {"type": "object", "properties": {}})
                    }
                }
            })

        return {"toolConfig": {"tools": bedrock_tools}}

    def create(self, model=None, messages=None, temperature=0.7,
               stream=False, tools=None, response_format=None,
               max_tokens=4096, timeout=None, **kwargs):
        """OpenAI 兼容的 create 方法"""
        model = model or self._default_model
        system_prompts, bedrock_messages = self._convert_messages(messages or [])

        params = {
            "modelId": model,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            }
        }

        if system_prompts:
            params["system"] = system_prompts

        tool_config = self._convert_tools(tools)
        if tool_config:
            params.update(tool_config)

        if stream:
            return self._stream_response(params)
        else:
            return self._sync_response(params)

    def _sync_response(self, params):
        """同步调用 Bedrock Converse"""
        response = self._client.converse(**params)

        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        usage_data = response.get("usage", {})

        text_content = ""
        tool_calls = []

        for block in content_blocks:
            if "text" in block:
                text_content += block["text"]
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(_ToolCall(
                    id=tu.get("toolUseId", ""),
                    type="function",
                    function=_Function(
                        name=tu.get("name", ""),
                        arguments=json.dumps(tu.get("input", {}))
                    )
                ))

        finish_reason = "tool_calls" if tool_calls else "stop"
        msg = _Message(
            role="assistant",
            content=text_content,
            tool_calls=tool_calls if tool_calls else None
        )

        return _CompletionResponse(
            choices=[_Choice(message=msg, finish_reason=finish_reason)],
            usage=_Usage(
                prompt_tokens=usage_data.get("inputTokens", 0),
                completion_tokens=usage_data.get("outputTokens", 0),
                total_tokens=usage_data.get("inputTokens", 0) + usage_data.get("outputTokens", 0)
            )
        )

    def _stream_response(self, params):
        """流式调用 Bedrock ConverseStream"""
        response = self._client.converse_stream(**params)
        return BedrockStreamIterator(response.get("stream", []))


class BedrockStreamIterator:
    """将 Bedrock 流式事件转为 OpenAI 兼容的 chunk 格式"""

    def __init__(self, event_stream):
        self._stream = event_stream
        self.usage = None  # 流结束后可读取

    def __iter__(self):
        for event in self._stream:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    yield _StreamChunk(choices=[
                        _Choice(delta=_Delta(content=delta["text"]), finish_reason=None)
                    ])

            elif "contentBlockStart" in event:
                start = event["contentBlockStart"].get("start", {})
                if "toolUse" in start:
                    tu = start["toolUse"]
                    tc = _ToolCall(
                        id=tu.get("toolUseId", ""),
                        type="function",
                        function=_Function(name=tu.get("name", ""), arguments="")
                    )
                    yield _StreamChunk(choices=[
                        _Choice(delta=_Delta(tool_calls=[tc]), finish_reason=None)
                    ])

            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason", "end_turn")
                finish = "tool_calls" if stop_reason == "tool_use" else "stop"
                yield _StreamChunk(choices=[
                    _Choice(delta=_Delta(), finish_reason=finish)
                ])

            elif "metadata" in event:
                meta_usage = event["metadata"].get("usage", {})
                if meta_usage:
                    self.usage = _Usage(
                        prompt_tokens=meta_usage.get("inputTokens", 0),
                        completion_tokens=meta_usage.get("outputTokens", 0),
                        total_tokens=meta_usage.get("inputTokens", 0) + meta_usage.get("outputTokens", 0)
                    )


class _BedrockChat:
    """模拟 openai.chat 命名空间"""
    def __init__(self, completions):
        self.completions = completions


class BedrockOpenAIClient:
    """
    OpenAI SDK 兼容的 AWS Bedrock 客户端

    用法：
        client = BedrockOpenAIClient(...)
        # 完全像 OpenAI client 一样使用：
        response = client.chat.completions.create(model=..., messages=..., ...)
    """

    def __init__(self, aws_access_key_id: str, aws_secret_access_key: str,
                 region: str = 'us-east-1', model: str = 'anthropic.claude-sonnet-4-20250514'):
        import boto3
        self._bedrock = boto3.client(
            'bedrock-runtime',
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        completions = BedrockChatCompletions(self._bedrock, model)
        self.chat = _BedrockChat(completions)


# ===========================================
# GLM 兼容层
# ===========================================
# 包装 GLM 的 OpenAI 客户端，处理 response_format 等参数兼容性

class _GLMCompletions:
    """包装 chat.completions，处理 GLM 不支持的参数"""

    def __init__(self, inner):
        self._inner = inner

    def create(self, **kwargs):
        try:
            return self._inner.create(**kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            # 如果 response_format 导致报错，去掉后重试
            if 'response_format' in kwargs and (
                'response_format' in err_msg
                or 'unsupported' in err_msg
                or 'invalid' in err_msg
            ):
                print(f"[GLM兼容层] response_format 不受支持，去掉后重试")
                kwargs.pop('response_format')
                return self._inner.create(**kwargs)
            raise


class _GLMChat:
    def __init__(self, completions):
        self.completions = completions


class GLMCompatibleClient:
    """
    GLM OpenAI 兼容客户端包装器

    处理 GLM API 与 OpenAI API 的差异：
    - response_format 参数：GLM 可能不支持，自动降级
    - 其余属性透传到原始客户端
    """

    def __init__(self, inner_client):
        self._inner = inner_client
        self.chat = _GLMChat(_GLMCompletions(inner_client.chat.completions))

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ===========================================
# 模型路由器
# ===========================================

class ModelRouter:
    """
    模型路由器 - 统一使用 GLM，Sonnet 仅作为备用

    策略：
    1. 优先使用 GLM（默认 glm-4.5）
    2. GLM 不可用时回退 Sonnet
    """

    def __init__(self, usage_tracker: UsageTracker):
        self.usage_tracker = usage_tracker

        # 初始化 Sonnet 客户端 (AWS Bedrock)
        self.sonnet_client = None
        self.sonnet_model = config.SONNET_MODEL_ID
        if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY:
            try:
                self.sonnet_client = BedrockOpenAIClient(
                    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                    region=config.AWS_REGION,
                    model=self.sonnet_model
                )
                print(f"[ModelRouter] Sonnet 客户端初始化成功 (model={self.sonnet_model})")
            except Exception as e:
                print(f"[ModelRouter] Sonnet 客户端初始化失败: {e}")

        # 初始化 GLM 客户端 (OpenAI 兼容接口，带兼容层)
        self.glm_client = None
        self.glm_model = config.GLM_MODEL
        if config.GLM_API_KEY:
            try:
                from openai import OpenAI
                raw_client = OpenAI(
                    api_key=config.GLM_API_KEY,
                    base_url=config.GLM_BASE_URL
                )
                self.glm_client = GLMCompatibleClient(raw_client)
                print(f"[ModelRouter] GLM 客户端初始化成功 (model={self.glm_model})")
            except Exception as e:
                print(f"[ModelRouter] GLM 客户端初始化失败: {e}")

        # 多模型梯队（共用同一个 client，只切换 model name）
        self.glm_model_plus = config.GLM_MODEL_PLUS    # 开场白、PlanningAgent
        self.glm_model_flash = config.GLM_MODEL_FLASH  # 轻量任务：简历拆分
        if self.glm_client:
            print(f"[ModelRouter] GLM 模型梯队: {self.glm_model}(核心) / {self.glm_model_flash}(轻量)")

        self.budget = config.SONNET_BUDGET_PER_USER

    def get_client_for_user(self, user_id: str) -> tuple:
        """
        根据用户返回合适的 (client, model, provider)

        当前策略：统一使用 GLM，Sonnet 仅作为备用

        Returns:
            (client, model_name, provider_name)
        """
        # 优先使用 GLM
        if self.glm_client:
            return self.glm_client, self.glm_model, "glm"

        # GLM 不可用时回退 Sonnet
        if self.sonnet_client:
            print(f"[ModelRouter] GLM 不可用，回退到 Sonnet")
            return self.sonnet_client, self.sonnet_model, "sonnet"

        raise RuntimeError("无可用的 LLM 模型（GLM 和 Sonnet 均未配置）")

    def record_usage(self, user_id: str, provider: str, model: str,
                     input_tokens: int, output_tokens: int):
        """记录一次调用的用量和费用"""
        if provider == "sonnet":
            cost = (input_tokens / 1000 * config.SONNET_INPUT_PRICE_PER_1K +
                    output_tokens / 1000 * config.SONNET_OUTPUT_PRICE_PER_1K)
        else:
            cost = 0  # GLM 成本可后续添加

        self.usage_tracker.record_usage(
            user_id=user_id or "anonymous",
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_rmb=cost
        )

        if cost > 0:
            print(f"[ModelRouter] 记录用量: {provider}/{model} in={input_tokens} out={output_tokens} cost=¥{cost:.6f}")
