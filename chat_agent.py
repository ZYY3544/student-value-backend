"""
===========================================
简历优化 Agent 模块 (Resume Optimization Agent)
===========================================
基于 HAY 评估结果，通过多轮对话帮助用户优化简历

核心能力：
1. 会话管理（内存存储，支持多用户并发）
2. 动态 Prompt 构建（注入评测上下文）
3. 简历结构化拆分（LLM 辅助）
4. 多阶段对话控制（开场 → 诊断 → 优化 → 总结）
5. SSE 流式输出
"""

import uuid
import json
import time
import threading
from datetime import datetime
from typing import Dict, Optional, List, Generator
from openai import OpenAI


# ===========================================
# 会话管理
# ===========================================

class SessionManager:
    """
    内存会话管理器

    每个会话存储：对话历史、评测上下文、当前阶段、简历段落等
    """

    def __init__(self, ttl_seconds: int = 3600):
        self._sessions: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def create_session(self, assessment_context: dict, resume_text: str) -> str:
        """创建新会话，返回 session_id"""
        session_id = str(uuid.uuid4())
        now = time.time()

        session = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "phase": "opening",  # opening → optimizing → summary
            "messages": [],      # 完整对话历史 [{"role": "...", "content": "..."}]
            "assessment_context": assessment_context,
            "resume_text": resume_text,
            "resume_sections": None,  # 结构化拆分后填充
        }

        with self._lock:
            self._sessions[session_id] = session

        print(f"[Agent] 创建会话 {session_id[:8]}...")
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """获取会话，不存在或已过期返回 None"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            # 检查过期
            if time.time() - session["updated_at"] > self._ttl:
                del self._sessions[session_id]
                print(f"[Agent] 会话 {session_id[:8]} 已过期，已清理")
                return None
            return session

    def update_session(self, session_id: str, updates: dict):
        """更新会话字段"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.update(updates)
                session["updated_at"] = time.time()

    def add_message(self, session_id: str, role: str, content: str):
        """追加一条消息到对话历史"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session["messages"].append({"role": role, "content": content})
                session["updated_at"] = time.time()

    def cleanup_expired(self):
        """清理所有过期会话"""
        now = time.time()
        with self._lock:
            expired = [sid for sid, s in self._sessions.items()
                       if now - s["updated_at"] > self._ttl]
            for sid in expired:
                del self._sessions[sid]
            if expired:
                print(f"[Agent] 清理了 {len(expired)} 个过期会话")


# ===========================================
# Prompt 构建器
# ===========================================

class PromptBuilder:
    """
    动态构建 System Prompt

    根据当前阶段和评测上下文，生成不同的 system prompt
    """

    @staticmethod
    def build_system_prompt(session: dict) -> str:
        """根据会话状态构建 system prompt"""
        phase = session["phase"]
        ctx = session["assessment_context"]
        resume_text = session["resume_text"]

        # 基础角色设定
        base_role = """你是一位资深的简历优化顾问，同时也是 HAY 岗位评估体系的专家。
你的使命是帮助用户基于 HAY 评估结果，针对性地优化简历，提升职场竞争力。

【你的核心能力】
- 精通 HAY 评估体系的 8 个因素（专业知识PK、管理知识MK、沟通Comm、思维环境TE、思维挑战TC、行动自由FTA、影响范围M、影响性质NI）
- 能够从简历文字中识别出哪些因素被低估，哪些信息缺失
- 能给出具体的、可操作的简历修改建议
- 语气友好专业，像一位有经验的职业导师

【关键原则】
- 所有建议必须基于用户的真实经历，绝不编造
- 区分"呈现不足"（简历没写好）和"能力不足"（确实缺乏经历）
- 对于呈现不足：帮用户重新组织语言，突出价值
- 对于能力不足：建议补充经历（实习/项目/竞赛），而非凭空捏造
- 每条建议要说清楚"改了之后，对评估的哪个维度有提升"
"""

        # 评测上下文注入
        context_section = PromptBuilder._build_context_section(ctx)

        # 简历内容（截断保护）
        resume_preview = resume_text[:3000] if len(resume_text) > 3000 else resume_text
        resume_section = f"""
【用户的简历内容】
{resume_preview}
"""

        # 阶段特定指令
        phase_instruction = PromptBuilder._build_phase_instruction(phase, ctx)

        return base_role + context_section + resume_section + phase_instruction

    @staticmethod
    def _build_context_section(ctx: dict) -> str:
        """构建评测上下文段落"""
        factors = ctx.get("factors", {})
        abilities = ctx.get("abilities", {})
        grade = ctx.get("grade", "未知")
        salary = ctx.get("salaryRange", "未知")
        job_title = ctx.get("jobTitle", "未知")
        job_function = ctx.get("jobFunction", "未知")

        # 构建因素描述
        factor_names = {
            "practical_knowledge": "专业知识(PK)",
            "managerial_knowledge": "管理知识(MK)",
            "communication": "沟通技巧(Comm)",
            "thinking_environment": "思维环境(TE)",
            "thinking_challenge": "思维挑战(TC)",
            "freedom_to_act": "行动自由(FTA)",
            "magnitude": "影响范围(M)",
            "nature_of_impact": "影响性质(NI)",
        }

        factors_text = "\n".join(
            f"  - {factor_names.get(k, k)}: {v}"
            for k, v in factors.items()
        )

        # 构建能力描述
        abilities_text = "\n".join(
            f"  - {name}: {info.get('score', '?')}分 ({info.get('level', '?')})"
            for name, info in abilities.items()
        ) if isinstance(abilities, dict) else "  暂无能力数据"

        return f"""
【用户的评测结果 - 这是你分析的基础】
- 目标岗位: {job_title}
- 所属职能: {job_function}
- 评估职级: {grade}
- 薪酬区间: {salary}

HAY 8因素档位:
{factors_text}

5维能力得分:
{abilities_text}
"""

    @staticmethod
    def _build_phase_instruction(phase: str, ctx: dict) -> str:
        """根据阶段构建指令"""

        if phase == "opening":
            # 找出最弱的能力维度
            abilities = ctx.get("abilities", {})
            weak_abilities = []
            if isinstance(abilities, dict):
                sorted_abs = sorted(abilities.items(),
                                    key=lambda x: x[1].get("score", 50))
                weak_abilities = [name for name, info in sorted_abs[:2]]

            weak_text = "、".join(weak_abilities) if weak_abilities else "部分维度"

            return f"""
【当前阶段：开场诊断】

你的任务是：
1. 简短问候用户（1句话）
2. 快速总结评测结果的亮点和短板（2-3句话），重点提及{weak_text}的提升空间
3. 提出 2-3 个你能帮忙优化的具体方向（列出来让用户选择），例如：
   - "实习/工作经历的描述可以更突出成果"
   - "项目经历缺少量化数据"
   - "可以补充一段能体现XX能力的经历"
4. 问用户想从哪里开始

【格式要求】
- 总字数控制在 200-300 字
- 语气温暖专业，不要过于正式
- 用 **双星号** 高亮关键词
- 不要使用 markdown 标题或列表符号，用自然语言组织
"""

        elif phase == "optimizing":
            return """
【当前阶段：优化建议】

你的任务是：
1. 针对用户提到的问题或选择的方向，给出具体建议
2. 如果是某段经历需要改写，给出修改前后的对比
3. 每条建议说明"这样改，可以提升哪个能力维度"
4. 一次聚焦 1-2 个点，不要一口气给太多
5. 改完一个点后，询问用户是否满意，以及要不要继续改其他部分

【格式要求】
- 修改建议用具体的文字示范，不要只说"你应该加上量化数据"
- 对比格式：先展示原文（标注问题），再展示改写版本
- 保持对话感，不要写成报告
- 语气鼓励为主

【重要】
- 你给出的改写必须基于简历中已有的信息进行润色和重组
- 如果需要用户补充信息（比如具体数字、项目成果），要主动追问
- 不要自己编造用户没提到的经历或数据
"""

        elif phase == "summary":
            return """
【当前阶段：总结回顾】

你的任务是：
1. 总结本次优化了哪些内容
2. 指出优化后预计能提升哪些能力维度
3. 给出 1-2 个后续建议（下次可以继续优化的方向，或者需要补充的经历）
4. 鼓励用户

【格式要求】
- 简洁明了，200字以内
- 正能量收尾
"""

        return ""


# ===========================================
# 简历结构化拆分
# ===========================================

def split_resume_sections(client: OpenAI, model: str, resume_text: str) -> List[dict]:
    """
    使用 LLM 将简历拆分为结构化段落

    Returns:
        [
            {"type": "education", "title": "教育经历", "content": "..."},
            {"type": "internship", "title": "字节跳动-产品实习", "content": "..."},
            {"type": "project", "title": "xxx项目", "content": "..."},
            ...
        ]
    """
    system_prompt = """你是一个简历解析专家。请将以下简历内容拆分为结构化的段落。

输出 JSON 数组，每个元素包含：
- type: 段落类型，取值为 education/internship/project/competition/skill/other
- title: 段落标题（简短描述，如"教育经历"、"字节跳动-产品实习"、"XX竞赛"）
- content: 该段落的原始内容

注意：
- 保持原文内容不变，只做拆分
- 如果简历中有多段实习经历，拆成多个段落
- 教育经历合并为一个段落
- 技能/证书合并为一个段落

输出纯 JSON，不要其他文字。"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": resume_text[:4000]}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        result = json.loads(content)

        # 兼容 LLM 返回 {"sections": [...]} 或直接 [...]
        if isinstance(result, dict):
            sections = result.get("sections", result.get("data", []))
        elif isinstance(result, list):
            sections = result
        else:
            sections = []

        print(f"[Agent] 简历拆分完成，共 {len(sections)} 个段落")
        return sections

    except Exception as e:
        print(f"[Agent] 简历拆分失败: {e}")
        # 降级：整体作为一个段落
        return [{"type": "other", "title": "完整简历", "content": resume_text}]


# ===========================================
# Chat Agent 核心类
# ===========================================

class ChatAgent:
    """
    简历优化对话 Agent

    使用方式：
        agent = ChatAgent(llm_service)
        session_id = agent.start_session(assessment_context, resume_text)
        response_stream = agent.chat(session_id, user_message)
    """

    def __init__(self, client: OpenAI, model: str = "deepseek-chat"):
        self.client = client
        self.model = model
        self.session_manager = SessionManager(ttl_seconds=3600)

    def start_session(self, assessment_context: dict, resume_text: str) -> dict:
        """
        开启新的对话会话

        Args:
            assessment_context: 评测结果（包含 factors, abilities, grade, salary 等）
            resume_text: 简历原文

        Returns:
            {"session_id": "...", "greeting": "..."}
        """
        session_id = self.session_manager.create_session(
            assessment_context=assessment_context,
            resume_text=resume_text
        )

        # 后台拆分简历（不阻塞开场）
        session = self.session_manager.get_session(session_id)
        if session:
            try:
                sections = split_resume_sections(self.client, self.model, resume_text)
                self.session_manager.update_session(session_id, {
                    "resume_sections": sections
                })
            except Exception as e:
                print(f"[Agent] 简历拆分失败（非阻塞）: {e}")

        # 生成开场白
        greeting = self._generate_opening(session_id)

        return {
            "session_id": session_id,
            "greeting": greeting
        }

    def _generate_opening(self, session_id: str) -> str:
        """生成开场白（非流式，用于 start 接口）"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return "会话创建失败，请重试。"

        system_prompt = PromptBuilder.build_system_prompt(session)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "你好，帮我看看简历怎么改"}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
            )
            greeting = response.choices[0].message.content.strip()

            # 保存到对话历史
            self.session_manager.add_message(session_id, "user", "你好，帮我看看简历怎么改")
            self.session_manager.add_message(session_id, "assistant", greeting)

            # 开场完成后进入优化阶段
            self.session_manager.update_session(session_id, {"phase": "optimizing"})

            return greeting
        except Exception as e:
            print(f"[Agent] 开场白生成失败: {e}")
            return "你好！我已经看过你的评测结果和简历了。准备好了就告诉我，我们可以开始优化简历。"

    def chat(self, session_id: str, user_message: str) -> Optional[str]:
        """
        处理用户消息，返回完整回复（非流式）

        Args:
            session_id: 会话ID
            user_message: 用户消息

        Returns:
            Agent 的回复文本，会话无效返回 None
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return None

        # 保存用户消息
        self.session_manager.add_message(session_id, "user", user_message)

        # 阶段控制：检查用户是否想结束
        if any(kw in user_message for kw in ["总结", "结束", "就这些", "谢谢", "没了", "差不多了"]):
            self.session_manager.update_session(session_id, {"phase": "summary"})

        # 重新获取更新后的 session
        session = self.session_manager.get_session(session_id)
        system_prompt = PromptBuilder.build_system_prompt(session)

        # 构建完整消息列表
        messages = [{"role": "system", "content": system_prompt}]

        # 对话历史（控制长度，保留最近 20 轮）
        history = session["messages"]
        if len(history) > 40:  # 每轮 user+assistant = 2条
            history = history[:2] + history[-38:]  # 保留开场 + 最近19轮

        messages.extend(history)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
            )
            reply = response.choices[0].message.content.strip()

            # 保存 assistant 回复
            self.session_manager.add_message(session_id, "assistant", reply)

            return reply
        except Exception as e:
            print(f"[Agent] 对话失败: {e}")
            return "抱歉，处理你的消息时遇到了问题，请再试一次。"

    def chat_stream(self, session_id: str, user_message: str) -> Generator[str, None, None]:
        """
        处理用户消息，返回流式回复（SSE 格式）

        Yields:
            逐块的文本内容
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            yield "[ERROR] 会话已过期或不存在"
            return

        # 保存用户消息
        self.session_manager.add_message(session_id, "user", user_message)

        # 阶段控制
        if any(kw in user_message for kw in ["总结", "结束", "就这些", "谢谢", "没了", "差不多了"]):
            self.session_manager.update_session(session_id, {"phase": "summary"})

        # 重新获取
        session = self.session_manager.get_session(session_id)
        system_prompt = PromptBuilder.build_system_prompt(session)

        messages = [{"role": "system", "content": system_prompt}]
        history = session["messages"]
        if len(history) > 40:
            history = history[:2] + history[-38:]
        messages.extend(history)

        full_reply = ""

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_reply += text
                    yield text

            # 保存完整回复
            if full_reply:
                self.session_manager.add_message(session_id, "assistant", full_reply)

        except Exception as e:
            print(f"[Agent] 流式对话失败: {e}")
            error_msg = "抱歉，处理消息时遇到了问题，请再试一次。"
            self.session_manager.add_message(session_id, "assistant", error_msg)
            yield error_msg

    def get_history(self, session_id: str) -> Optional[List[dict]]:
        """获取对话历史"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return None
        # 过滤掉 system 消息，只返回 user 和 assistant
        return [msg for msg in session["messages"]
                if msg["role"] in ("user", "assistant")]
