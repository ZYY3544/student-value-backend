"""
===========================================
模型路由器 (Model Router)
===========================================
Claude 模型通过 OpenRouter（OpenAI 兼容接口）调用。
GLM 作为备用。

包含：
- UsageTracker: 用量追踪（PostgreSQL）
- GLMCompatibleClient: GLM 兼容层
- ModelRouter: 模型选择路由器
"""

import os
from typing import Optional
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
# GLM 兼容层
# ===========================================

class _GLMCompletions:
    """包装 chat.completions，处理 GLM 不支持的参数"""

    def __init__(self, inner):
        self._inner = inner

    def create(self, **kwargs):
        try:
            return self._inner.create(**kwargs)
        except Exception as e:
            err_msg = str(e).lower()
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
    """GLM OpenAI 兼容客户端包装器"""

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
    模型路由器 — OpenRouter (Haiku) 为主力，GLM 备用

    策略：
    1. 优先使用 OpenRouter Haiku 4.5
    2. OpenRouter 不可用时回退 GLM
    3. Sonnet 仅用于报告解读（在 ChatAgent 中独立管理）
    """

    def __init__(self, usage_tracker: UsageTracker):
        self.usage_tracker = usage_tracker

        self.haiku_model = config.HAIKU_MODEL_ID
        self.sonnet_model = config.SONNET_MODEL_ID

        # OpenRouter 客户端（主力，Haiku 4.5）
        self.openrouter_client = None
        if config.OPENROUTER_API_KEY:
            try:
                from openai import OpenAI
                self.openrouter_client = OpenAI(
                    api_key=config.OPENROUTER_API_KEY,
                    base_url=config.OPENROUTER_BASE_URL,
                )
                print(f"[ModelRouter] OpenRouter 客户端初始化成功 (model={self.haiku_model})")
            except Exception as e:
                print(f"[ModelRouter] OpenRouter 客户端初始化失败: {e}")

        # GLM 客户端（备用）
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
                print(f"[ModelRouter] GLM 客户端初始化成功 (model={self.glm_model}，备用)")
            except Exception as e:
                print(f"[ModelRouter] GLM 客户端初始化失败: {e}")

        # 兼容属性
        self.glm_model_plus = self.haiku_model if self.openrouter_client else config.GLM_MODEL_PLUS
        self.glm_model_flash = self.haiku_model if self.openrouter_client else config.GLM_MODEL_FLASH

        if self.openrouter_client:
            print(f"[ModelRouter] 主力模型: Haiku 4.5 via OpenRouter ({self.haiku_model})")
        elif self.glm_client:
            print(f"[ModelRouter] 主力模型: GLM ({self.glm_model})")

        self.budget = config.SONNET_BUDGET_PER_USER

    def get_client_for_user(self, user_id: str) -> tuple:
        """
        返回 (client, model, provider)
        OpenRouter 和 GLM 都是 OpenAI 兼容接口，调用方式完全一致。
        """
        if self.openrouter_client:
            return self.openrouter_client, self.haiku_model, "openrouter"

        if self.glm_client:
            print(f"[ModelRouter] OpenRouter 不可用，回退到 GLM")
            return self.glm_client, self.glm_model, "glm"

        raise RuntimeError("无可用的 LLM 模型（OpenRouter 和 GLM 均未配置）")

    def record_usage(self, user_id: str, provider: str, model: str,
                     input_tokens: int, output_tokens: int):
        """记录一次调用的用量和费用"""
        if provider == "sonnet":
            cost = (input_tokens / 1000 * config.SONNET_INPUT_PRICE_PER_1K +
                    output_tokens / 1000 * config.SONNET_OUTPUT_PRICE_PER_1K)
        else:
            cost = 0

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
