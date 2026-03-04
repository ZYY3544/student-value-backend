"""
===========================================
简历优化 Agent 模块 (Resume Optimization Agent)
===========================================
基于 HAY 评估结果，通过多轮对话帮助用户优化简历

核心能力：
1. 会话管理（内存存储，支持多用户并发）
2. 动态 Prompt 构建（首尾强化 + 降噪排序）
3. 简历结构化拆分（LLM 辅助）
4. 多阶段对话控制（开场 → 诊断 → 优化 → 总结）
5. 结构化记忆（关键信息锚定，防止上下文腐烂）
6. 对话历史压缩与降噪
7. SSE 流式输出
"""

import uuid
import json
import time
import threading
from datetime import datetime
from typing import Dict, Optional, List, Generator
from openai import OpenAI


# ===========================================
# 结构化记忆
# ===========================================

class ConversationMemory:
    """
    结构化记忆管理器

    从对话流中提取关键信息，独立存储，每轮注入 prompt。
    对话历史可以压缩/丢弃，但结构化记忆不会丢失。
    """

    def __init__(self):
        self.resume_summary: str = ""           # 简历核心摘要
        self.assessment_highlights: str = ""     # 评测亮点
        self.assessment_weaknesses: str = ""     # 评测短板
        self.agreed_modifications: List[str] = []  # 已达成的修改共识
        self.user_preferences: Dict[str, str] = {} # 用户偏好（意向岗位、关注方向等）
        self.optimized_sections: List[str] = []    # 已优化过的段落
        self.pending_questions: List[str] = []     # 待用户补充的信息

    def add_agreed_modification(self, modification: str):
        """记录一条已达成的修改共识"""
        if modification not in self.agreed_modifications:
            self.agreed_modifications.append(modification)

    def add_optimized_section(self, section: str):
        """记录已优化的段落"""
        if section not in self.optimized_sections:
            self.optimized_sections.append(section)

    def set_user_preference(self, key: str, value: str):
        """记录用户偏好"""
        self.user_preferences[key] = value

    def to_context_string(self) -> str:
        """将结构化记忆转为可注入 prompt 的文本"""
        parts = []

        if self.agreed_modifications:
            mods = "\n".join(f"  {i+1}. {m}" for i, m in enumerate(self.agreed_modifications))
            parts.append(f"【本次对话已达成的修改共识】\n{mods}")

        if self.optimized_sections:
            sections = "、".join(self.optimized_sections)
            parts.append(f"【已优化过的段落】{sections}")

        if self.user_preferences:
            prefs = "\n".join(f"  - {k}: {v}" for k, v in self.user_preferences.items())
            parts.append(f"【用户偏好】\n{prefs}")

        if self.pending_questions:
            qs = "\n".join(f"  - {q}" for q in self.pending_questions)
            parts.append(f"【待用户补充的信息】\n{qs}")

        return "\n\n".join(parts)

    def has_content(self) -> bool:
        return bool(self.agreed_modifications or self.optimized_sections
                     or self.user_preferences or self.pending_questions)


# ===========================================
# 会话管理
# ===========================================

class SessionManager:
    """
    内存会话管理器

    每个会话存储：对话历史、评测上下文、当前阶段、简历段落、结构化记忆等
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
            "memory": ConversationMemory(),  # 结构化记忆
            "compressed_history": "",  # 压缩后的早期对话摘要
            "message_count": 0,  # 总消息计数（用于判断是否需要压缩）
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
                session["message_count"] += 1
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
# 对话历史压缩与降噪
# ===========================================

class HistoryCompressor:
    """
    对话历史压缩与降噪

    策略：
    1. 降噪：过滤掉闲聊、重复、低信息量的对话
    2. 压缩：当对话超过阈值时，用 LLM 将早期对话压缩为摘要
    3. 排序：保证最近的对话原样保留（高注意力区域）
    """

    # 需要压缩的消息数阈值（超过此数才触发压缩）
    COMPRESS_THRESHOLD = 16  # 8轮对话

    # 保留最近多少条消息不压缩
    KEEP_RECENT = 8  # 最近4轮原始保留

    @staticmethod
    def should_compress(session: dict) -> bool:
        """判断是否需要压缩"""
        return len(session["messages"]) > HistoryCompressor.COMPRESS_THRESHOLD

    @staticmethod
    def compress_history(client: OpenAI, model: str, session: dict) -> str:
        """
        用 LLM 将早期对话压缩为摘要

        Returns:
            压缩后的摘要文本
        """
        messages = session["messages"]
        keep_recent = HistoryCompressor.KEEP_RECENT

        # 需要压缩的部分：除了最近 keep_recent 条之外的早期消息
        to_compress = messages[:-keep_recent]

        if not to_compress:
            return session.get("compressed_history", "")

        # 拼接已有摘要 + 新的需要压缩的对话
        existing_summary = session.get("compressed_history", "")
        conversation_text = ""
        if existing_summary:
            conversation_text += f"【之前的对话摘要】\n{existing_summary}\n\n【新的对话内容】\n"

        for msg in to_compress:
            role_name = "用户" if msg["role"] == "user" else "顾问"
            conversation_text += f"{role_name}: {msg['content']}\n\n"

        compress_prompt = """请将以下对话内容压缩为简洁的摘要，保留以下关键信息：
1. 用户关心的核心问题和优化方向
2. 已经给出的具体修改建议（保留修改前后对比的要点）
3. 用户表达的偏好和意向
4. 已达成的共识和待确认的事项

要求：
- 用第三人称客观描述
- 200-400字以内
- 按时间顺序组织
- 不要遗漏任何已达成的修改共识"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": compress_prompt},
                    {"role": "user", "content": conversation_text}
                ],
                temperature=0.0,
            )
            summary = response.choices[0].message.content.strip()
            print(f"[Agent] 对话历史压缩完成，原 {len(to_compress)} 条消息 → 摘要")
            return summary
        except Exception as e:
            print(f"[Agent] 对话历史压缩失败: {e}")
            return existing_summary

    @staticmethod
    def extract_memory(client: OpenAI, model: str, session: dict):
        """
        从最近的对话中提取结构化记忆

        在每轮对话后调用，将关键信息提取到 ConversationMemory 中
        """
        memory: ConversationMemory = session["memory"]
        messages = session["messages"]

        # 只看最近2条消息（最后一轮对话）
        if len(messages) < 2:
            return

        recent = messages[-2:]  # user + assistant
        recent_text = "\n".join(f"{'用户' if m['role']=='user' else '顾问'}: {m['content']}" for m in recent)

        extract_prompt = """分析以下对话，提取结构化信息。如果某项信息不存在，对应字段留空字符串。

输出严格的 JSON 格式：
{
    "agreed_modification": "本轮达成的修改共识（如果有的话）",
    "optimized_section": "本轮优化了哪个段落（如果有的话）",
    "user_preference": {"key": "偏好类别", "value": "偏好内容"},
    "pending_question": "需要用户后续补充的信息（如果有的话）"
}

注意：只提取明确出现的信息，不要推测。如果本轮没有相关信息，对应字段留空字符串。"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": extract_prompt},
                    {"role": "user", "content": recent_text}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            if result.get("agreed_modification"):
                memory.add_agreed_modification(result["agreed_modification"])
            if result.get("optimized_section"):
                memory.add_optimized_section(result["optimized_section"])
            if result.get("user_preference") and isinstance(result["user_preference"], dict):
                pref = result["user_preference"]
                if pref.get("key") and pref.get("value"):
                    memory.set_user_preference(pref["key"], pref["value"])
            if result.get("pending_question"):
                memory.pending_questions = [result["pending_question"]]  # 只保留最新的

        except Exception as e:
            print(f"[Agent] 记忆提取失败（非阻塞）: {e}")


# ===========================================
# Prompt 构建器（首尾强化 + 降噪排序）
# ===========================================

class PromptBuilder:
    """
    动态构建 Prompt（遵循长上下文最佳实践）

    结构（从上到下）：
    ┌─────────────────────────────┐
    │ System Prompt（首部高注意力） │
    │ = 核心身份 + 行为规则        │
    ├─────────────────────────────┤
    │ <history>                    │
    │ 压缩摘要 + 最近原始对话      │
    │ </history>                   │
    ├─────────────────────────────┤
    │ <context>                    │
    │ 评测结果 + 简历 + 结构化记忆  │
    │ </context>                   │
    ├─────────────────────────────┤
    │ <task>（尾部高注意力）        │
    │ 当前阶段精简指令 + 核心提醒   │
    │ </task>                      │
    └─────────────────────────────┘
    """

    @staticmethod
    def build_system_prompt(session: dict) -> str:
        """
        构建 system prompt（首部）

        只放核心身份和行为规则，利用首部高注意力权重建立任务基调
        """
        return """你是一位资深的简历优化顾问，同时也是 HAY 岗位评估体系的专家。
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

    @staticmethod
    def build_user_prompt(session: dict) -> str:
        """
        构建注入到最新 user message 前的上下文块

        结构：<history> + <context> + <task>
        利用尾部高注意力权重，把当前阶段指令放在最后
        """
        phase = session["phase"]
        ctx = session["assessment_context"]
        resume_text = session["resume_text"]
        memory: ConversationMemory = session["memory"]
        compressed_history = session.get("compressed_history", "")

        parts = []

        # === <history> 压缩后的早期对话摘要 ===
        if compressed_history:
            parts.append(f"""<history>
【早期对话摘要】
{compressed_history}
</history>""")

        # === <context> 评测结果 + 简历 + 结构化记忆 ===
        context_section = PromptBuilder._build_context_section(ctx, resume_text, memory)
        parts.append(f"""<context>
{context_section}
</context>""")

        # === <task> 当前阶段精简指令（尾部高注意力） ===
        task_section = PromptBuilder._build_task_section(phase, ctx, memory)
        parts.append(f"""<task>
{task_section}
</task>""")

        return "\n\n".join(parts)

    @staticmethod
    def _build_context_section(ctx: dict, resume_text: str, memory: ConversationMemory) -> str:
        """构建 <context> 块：评测结果 + 简历 + 结构化记忆"""
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

        # 简历内容（截断保护）
        resume_preview = resume_text[:3000] if len(resume_text) > 3000 else resume_text

        section = f"""【用户的评测结果 - 这是你分析的基础】
- 目标岗位: {job_title}
- 所属职能: {job_function}
- 评估职级: {grade}
- 薪酬区间: {salary}

HAY 8因素档位:
{factors_text}

5维能力得分:
{abilities_text}

【用户的简历内容】
{resume_preview}"""

        # 结构化记忆（如果有的话）
        if memory.has_content():
            section += f"\n\n{memory.to_context_string()}"

        return section

    @staticmethod
    def _build_task_section(phase: str, ctx: dict, memory: ConversationMemory) -> str:
        """
        构建 <task> 块：当前阶段精简指令

        放在 prompt 最尾部，利用尾部高注意力权重防止阶段漂移
        """

        if phase == "opening":
            # 找出最弱的能力维度
            abilities = ctx.get("abilities", {})
            weak_abilities = []
            if isinstance(abilities, dict):
                sorted_abs = sorted(abilities.items(),
                                    key=lambda x: x[1].get("score", 50))
                weak_abilities = [name for name, info in sorted_abs[:2]]

            weak_text = "、".join(weak_abilities) if weak_abilities else "部分维度"

            return f"""【当前阶段：开场诊断】

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

⚠️ 核心提醒：你现在处于【开场诊断】阶段，只做诊断和方向推荐，不要直接给出具体的修改建议。"""

        elif phase == "optimizing":
            # 动态注入已优化的段落信息，避免重复
            avoid_text = ""
            if memory.optimized_sections:
                sections = "、".join(memory.optimized_sections)
                avoid_text = f"\n- 已优化过的段落（{sections}）不要重复优化，除非用户主动要求"

            return f"""【当前阶段：优化建议】

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
- 不要自己编造用户没提到的经历或数据{avoid_text}

⚠️ 核心提醒：你现在处于【优化建议】阶段，请专注于给出具体可操作的修改建议。"""

        elif phase == "summary":
            return """【当前阶段：总结回顾】

你的任务是：
1. 总结本次优化了哪些内容
2. 指出优化后预计能提升哪些能力维度
3. 给出 1-2 个后续建议（下次可以继续优化的方向，或者需要补充的经历）
4. 鼓励用户

【格式要求】
- 简洁明了，200字以内
- 正能量收尾

⚠️ 核心提醒：你现在处于【总结回顾】阶段，请做最终总结，不要再引入新的优化点。"""

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
        user_context = PromptBuilder.build_user_prompt(session)

        # 首尾强化：system prompt 在首部，task 指令在尾部（user message 中）
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context + "\n\n用户说：你好，帮我看看简历怎么改"}
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

    def _prepare_messages(self, session: dict) -> list:
        """
        构建发送给 LLM 的消息列表

        遵循首尾强化原则：
        - 首部：system prompt（核心身份 + 行为规则）
        - 中部：压缩摘要 + 最近对话历史
        - 尾部：context + task 指令（通过 user prompt 注入）

        遵循降噪排序原则：
        - 早期对话压缩为摘要，减少噪音
        - 最近对话原样保留，保持精确性
        """
        # 首部：核心身份（高注意力区域）
        system_prompt = PromptBuilder.build_system_prompt(session)
        messages = [{"role": "system", "content": system_prompt}]

        # 中部：对话历史（降噪处理后）
        history = session["messages"]
        compressed = session.get("compressed_history", "")

        if compressed:
            # 有压缩摘要时，只保留最近几轮原始对话
            recent_history = history[-HistoryCompressor.KEEP_RECENT:]
        elif len(history) > HistoryCompressor.COMPRESS_THRESHOLD:
            # 还没压缩但已经很长，临时截断
            recent_history = history[:2] + history[-(HistoryCompressor.KEEP_RECENT):]
        else:
            # 对话还不长，全部保留
            recent_history = history

        messages.extend(recent_history)

        # 尾部：context + task 指令（高注意力区域）
        # 作为最后一条 user message 的前缀注入
        user_context = PromptBuilder.build_user_prompt(session)

        # 把 context+task 注入到最后一条 user message
        if messages and messages[-1]["role"] == "user":
            original_msg = messages[-1]["content"]
            messages[-1] = {
                "role": "user",
                "content": f"{user_context}\n\n用户说：{original_msg}"
            }
        else:
            # 安全兜底
            messages.append({"role": "user", "content": user_context})

        return messages

    def _post_chat_processing(self, session_id: str):
        """
        每轮对话后的后处理

        1. 提取结构化记忆
        2. 判断是否需要压缩对话历史
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return

        # 1. 提取结构化记忆（异步，不阻塞响应）
        try:
            HistoryCompressor.extract_memory(self.client, self.model, session)
        except Exception as e:
            print(f"[Agent] 记忆提取失败: {e}")

        # 2. 判断是否需要压缩
        if HistoryCompressor.should_compress(session):
            try:
                compressed = HistoryCompressor.compress_history(
                    self.client, self.model, session
                )
                # 更新摘要，并裁剪消息列表
                keep_recent = HistoryCompressor.KEEP_RECENT
                recent_messages = session["messages"][-keep_recent:]
                self.session_manager.update_session(session_id, {
                    "compressed_history": compressed,
                    "messages": recent_messages,
                })
                print(f"[Agent] 会话 {session_id[:8]} 历史已压缩，保留最近 {len(recent_messages)} 条消息")
            except Exception as e:
                print(f"[Agent] 历史压缩失败: {e}")

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

        # 构建消息（首尾强化 + 降噪排序）
        messages = self._prepare_messages(session)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
            )
            reply = response.choices[0].message.content.strip()

            # 保存 assistant 回复
            self.session_manager.add_message(session_id, "assistant", reply)

            # 后处理：提取记忆 + 判断压缩
            self._post_chat_processing(session_id)

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

        # 构建消息（首尾强化 + 降噪排序）
        messages = self._prepare_messages(session)

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

            # 后处理：提取记忆 + 判断压缩
            self._post_chat_processing(session_id)

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
