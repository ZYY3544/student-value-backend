"""
===========================================
简历优化多 Agent 系统 (Multi-Agent Resume Optimizer)
===========================================
基于 HAY 评估结果，通过多 Agent 协作帮助用户优化简历

架构：
┌──────────────────────────────────────────┐
│  ChatAgent（协调者/Orchestrator）          │
│  职责：会话管理、阶段路由、上下文传递       │
└──────┬──────────┬──────────┬─────────────┘
       │          │          │
  DiagnosisAgent  OptimizeAgent  ReportAgent
  (开场诊断)       (简历优化)     (总结报告)
  专精HAY分析     专精段落改写    专精报告生成
  单轮·低温度     短轮·中温度    单轮·低温度

每个子 Agent 的优势：
1. Prompt 极其详尽（不用担心互相稀释）
2. 每次调用上下文干净（不存在上下文腐烂）
3. 可独立调参（温度、模型、超时等）

核心模块：
1. ConversationMemory - 结构化记忆（跨 Agent 共享）
2. SessionManager - 会话管理
3. HistoryCompressor - 对话历史压缩
4. ChatAgent - 协调者（对外接口不变）
"""

import uuid
import json
import time
import threading
from datetime import datetime
from typing import Dict, Optional, List, Generator
from openai import OpenAI

from multi_agent import DiagnosisAgent, OptimizeAgent, ReportAgent
from tool_executor import ToolExecutor


# ===========================================
# 结构化记忆
# ===========================================

class ConversationMemory:
    """
    结构化记忆管理器

    从对话流中提取关键信息，独立存储，每轮注入子 Agent 的上下文。
    对话历史可以压缩/丢弃，但结构化记忆不会丢失。
    这是跨 Agent 共享的状态——所有子 Agent 都能看到。
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
            parts.append(f"【已优化过的段落（不要重复优化，除非用户要求）】{sections}")

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
            "memory": ConversationMemory(),  # 结构化记忆（跨 Agent 共享）
            "compressed_history": "",  # 压缩后的早期对话摘要
            "message_count": 0,  # 总消息计数（用于判断是否需要压缩）
        }

        with self._lock:
            self._sessions[session_id] = session

        print(f"[Orchestrator] 创建会话 {session_id[:8]}...")
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
                print(f"[Orchestrator] 会话 {session_id[:8]} 已过期，已清理")
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
                print(f"[Orchestrator] 清理了 {len(expired)} 个过期会话")


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
            print(f"[Orchestrator] 对话历史压缩完成，原 {len(to_compress)} 条消息 → 摘要")
            return summary
        except Exception as e:
            print(f"[Orchestrator] 对话历史压缩失败: {e}")
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
            print(f"[Orchestrator] 记忆提取失败（非阻塞）: {e}")


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

        print(f"[Orchestrator] 简历拆分完成，共 {len(sections)} 个段落")
        return sections

    except Exception as e:
        print(f"[Orchestrator] 简历拆分失败: {e}")
        # 降级：整体作为一个段落
        return [{"type": "other", "title": "完整简历", "content": resume_text}]


# ===========================================
# Chat Agent（协调者 / Orchestrator）
# ===========================================

class ChatAgent:
    """
    简历优化多 Agent 协调者

    职责极简：
    1. 管理会话生命周期
    2. 根据阶段路由到对应的子 Agent
    3. 传递上下文（评测结果、简历、记忆、对话摘要）
    4. 后处理（记忆提取、历史压缩）

    对外接口不变：
        agent = ChatAgent(client, model)
        result = agent.start_session(assessment_context, resume_text)
        reply = agent.chat(session_id, user_message)
        stream = agent.chat_stream(session_id, user_message)
        history = agent.get_history(session_id)
    """

    # 触发总结阶段的关键词（仅当消息以这些词为主要意图时才触发）
    SUMMARY_KEYWORDS = ["总结", "结束", "就这些", "没了", "差不多了", "先这样", "可以了", "到此为止", "OK了", "好了就这些"]

    # 表示还想继续优化的信号词（优先级高于总结关键词）
    CONTINUE_KEYWORDS = ["继续", "还有", "再改", "帮我改", "下一", "另外", "还想", "接着", "补充", "优化"]

    def __init__(self, client: OpenAI, model: str = "deepseek-chat",
                 llm_service=None, convergence_engine=None):
        self.client = client
        self.model = model
        self.llm_service = llm_service
        self.convergence_engine = convergence_engine
        self.session_manager = SessionManager(ttl_seconds=3600)

        # 初始化子 Agent（每个 Agent 拥有独立的 Prompt，共享 LLM client）
        self.diagnosis_agent = DiagnosisAgent(client, model)
        # OptimizeAgent 的 tool_executor 在 start_session 时按会话创建
        self.optimize_agent = OptimizeAgent(client, model)
        self.report_agent = ReportAgent(client, model)

        print("[Orchestrator] 多 Agent 系统初始化完成")
        print(f"  - DiagnosisAgent: 开场诊断（温度 {DiagnosisAgent.TEMPERATURE}）")
        print(f"  - OptimizeAgent:  简历优化（温度 {OptimizeAgent.TEMPERATURE}）")
        print(f"  - ReportAgent:    总结报告（温度 {ReportAgent.TEMPERATURE}）")
        if llm_service and convergence_engine:
            print(f"  - ToolExecutor:   Function Call 工具调用已启用")

    def start_session(self, assessment_context: dict, resume_text: str) -> dict:
        """
        开启新的对话会话

        流程：
        1. 创建会话
        2. 调用 DiagnosisAgent 生成开场诊断
        3. 后台拆分简历结构
        4. 进入优化阶段

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

        # 为本次会话创建 ToolExecutor 并绑定到 OptimizeAgent
        if self.llm_service and self.convergence_engine:
            session = self.session_manager.get_session(session_id)
            tool_executor = ToolExecutor(
                llm_service=self.llm_service,
                convergence_engine=self.convergence_engine,
                conversation_memory=session["memory"],
            )
            # 缓存原始评测结果（含 resume_text），供工具使用
            tool_executor.set_original_assessment({
                **assessment_context,
                "resume_text": resume_text,
            })
            # 存到 session 中，后续 chat/chat_stream 可取用
            self.session_manager.update_session(session_id, {
                "tool_executor": tool_executor,
            })

        # 后台线程拆分简历（真正不阻塞开场）
        def _bg_split():
            try:
                sections = split_resume_sections(self.client, self.model, resume_text)
                self.session_manager.update_session(session_id, {
                    "resume_sections": sections
                })
            except Exception as e:
                print(f"[Orchestrator] 简历拆分失败（后台）: {e}")

        threading.Thread(target=_bg_split, daemon=True).start()

        # ===== 调用 DiagnosisAgent 生成开场白 =====
        print(f"[Orchestrator] 路由 → DiagnosisAgent（开场诊断）")
        greeting = self.diagnosis_agent.diagnose(
            assessment_context=assessment_context,
            resume_text=resume_text
        )

        # 保存到对话历史
        self.session_manager.add_message(session_id, "user", "你好，帮我看看简历怎么改")
        self.session_manager.add_message(session_id, "assistant", greeting)

        # 开场完成后进入优化阶段
        self.session_manager.update_session(session_id, {"phase": "optimizing"})

        return {
            "session_id": session_id,
            "greeting": greeting
        }

    def _get_agent_context(self, session: dict) -> dict:
        """
        从 session 中提取子 Agent 需要的上下文

        这是协调者的核心工作：把正确的上下文传给正确的 Agent
        """
        memory: ConversationMemory = session["memory"]
        messages = session["messages"]
        compressed = session.get("compressed_history", "")

        # 构建对话摘要：压缩历史 + 结构化记忆
        conversation_summary = ""
        if compressed:
            conversation_summary = compressed

        # 结构化记忆
        memory_context = memory.to_context_string() if memory.has_content() else ""

        # 最近的原始对话（保持上下文连贯性）
        # 多 Agent 架构下，子 Agent 不需要看全部历史
        # 只需要最近几轮 + 压缩摘要就够了
        recent_messages = messages[-HistoryCompressor.KEEP_RECENT:] if messages else []

        return {
            "assessment_context": session["assessment_context"],
            "resume_text": session["resume_text"],
            "conversation_summary": conversation_summary,
            "recent_messages": recent_messages,
            "memory_context": memory_context,
            "tool_executor": session.get("tool_executor"),
        }

    def _detect_phase_transition(self, user_message: str, current_phase: str) -> str:
        """
        检测阶段转换

        规则：
        1. optimizing → summary：命中总结关键词且不含继续优化信号
        2. summary → optimizing：用户在总结后又想继续优化（回退）
        3. 短消息优先（"谢谢帮我继续改" 含两类关键词时，继续信号优先）
        """
        msg = user_message.strip()

        # 检测是否有继续优化的意图
        has_continue = any(kw in msg for kw in self.CONTINUE_KEYWORDS)
        has_summary = any(kw in msg for kw in self.SUMMARY_KEYWORDS)

        if current_phase == "optimizing":
            # 有总结意图且没有继续意图 → 进入总结
            if has_summary and not has_continue:
                # 额外检查：消息太长（>30字）可能只是聊天中偶然提到关键词
                if len(msg) > 30 and not msg.endswith(("吧", "了", "。")):
                    return current_phase
                return "summary"

        elif current_phase == "summary":
            # 总结阶段后用户想继续优化 → 回退
            if has_continue:
                print(f"[Orchestrator] 用户在总结后继续优化，回退到 optimizing")
                return "optimizing"

        return current_phase

    def _post_chat_processing(self, session_id: str):
        """
        每轮对话后的后处理（后台线程执行，不阻塞用户）

        1. 提取结构化记忆（供下一轮子 Agent 使用）
        2. 判断是否需要压缩对话历史
        """
        def _bg_process():
            session = self.session_manager.get_session(session_id)
            if not session:
                return

            # 1. 提取结构化记忆
            try:
                HistoryCompressor.extract_memory(self.client, self.model, session)
            except Exception as e:
                print(f"[Orchestrator] 记忆提取失败: {e}")

            # 2. 判断是否需要压缩
            if HistoryCompressor.should_compress(session):
                try:
                    compressed = HistoryCompressor.compress_history(
                        self.client, self.model, session
                    )
                    keep_recent = HistoryCompressor.KEEP_RECENT
                    recent_messages = session["messages"][-keep_recent:]
                    self.session_manager.update_session(session_id, {
                        "compressed_history": compressed,
                        "messages": recent_messages,
                    })
                    print(f"[Orchestrator] 会话 {session_id[:8]} 历史已压缩，保留最近 {len(recent_messages)} 条消息")
                except Exception as e:
                    print(f"[Orchestrator] 历史压缩失败: {e}")

        threading.Thread(target=_bg_process, daemon=True).start()

    def chat(self, session_id: str, user_message: str) -> Optional[str]:
        """
        处理用户消息，返回完整回复（非流式）

        协调者流程：
        1. 保存用户消息
        2. 检测阶段转换
        3. 路由到对应子 Agent
        4. 保存回复
        5. 后处理

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

        # 阶段转换检测
        new_phase = self._detect_phase_transition(user_message, session["phase"])
        if new_phase != session["phase"]:
            print(f"[Orchestrator] 阶段转换: {session['phase']} → {new_phase}")
            self.session_manager.update_session(session_id, {"phase": new_phase})

        # 重新获取更新后的 session
        session = self.session_manager.get_session(session_id)
        phase = session["phase"]

        # 提取子 Agent 需要的上下文
        ctx = self._get_agent_context(session)

        # ===== 路由到对应的子 Agent =====
        try:
            if phase == "optimizing":
                print(f"[Orchestrator] 路由 → OptimizeAgent")
                # 绑定当前会话的 tool_executor
                self.optimize_agent.tool_executor = ctx.get("tool_executor")
                reply = self.optimize_agent.optimize_sync(
                    assessment_context=ctx["assessment_context"],
                    resume_text=ctx["resume_text"],
                    user_message=user_message,
                    conversation_summary=ctx["conversation_summary"],
                    recent_messages=ctx["recent_messages"],
                    memory_context=ctx["memory_context"],
                )
            elif phase == "summary":
                print(f"[Orchestrator] 路由 → ReportAgent")
                reply = self.report_agent.generate_report_sync(
                    assessment_context=ctx["assessment_context"],
                    resume_text=ctx["resume_text"],
                    conversation_summary=ctx["conversation_summary"],
                    memory_context=ctx["memory_context"],
                    recent_messages=ctx["recent_messages"],
                )
            else:
                # opening 阶段不应该走到这里（start_session 已处理）
                # 兜底：用 OptimizeAgent
                print(f"[Orchestrator] 兜底路由 → OptimizeAgent（阶段: {phase}）")
                self.optimize_agent.tool_executor = ctx.get("tool_executor")
                reply = self.optimize_agent.optimize_sync(
                    assessment_context=ctx["assessment_context"],
                    resume_text=ctx["resume_text"],
                    user_message=user_message,
                    conversation_summary=ctx["conversation_summary"],
                    recent_messages=ctx["recent_messages"],
                    memory_context=ctx["memory_context"],
                )

            # 保存回复
            self.session_manager.add_message(session_id, "assistant", reply)

            # 后处理
            self._post_chat_processing(session_id)

            return reply

        except Exception as e:
            print(f"[Orchestrator] 对话失败: {e}")
            return "抱歉，处理你的消息时遇到了问题，请再试一次。"

    def chat_stream(self, session_id: str, user_message: str) -> Generator[str, None, None]:
        """
        处理用户消息，返回流式回复（SSE 格式）

        协调者流程（与 chat 相同，但子 Agent 返回流式输出）：
        1. 保存用户消息
        2. 检测阶段转换
        3. 路由到对应子 Agent（流式）
        4. 收集完整回复并保存
        5. 后处理

        Yields:
            逐块的文本内容
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            yield "[ERROR] 会话已过期或不存在"
            return

        # 保存用户消息
        self.session_manager.add_message(session_id, "user", user_message)

        # 阶段转换检测
        new_phase = self._detect_phase_transition(user_message, session["phase"])
        if new_phase != session["phase"]:
            print(f"[Orchestrator] 阶段转换: {session['phase']} → {new_phase}")
            self.session_manager.update_session(session_id, {"phase": new_phase})

        # 重新获取
        session = self.session_manager.get_session(session_id)
        phase = session["phase"]

        # 提取子 Agent 需要的上下文
        ctx = self._get_agent_context(session)

        full_reply = ""

        try:
            # ===== 路由到对应的子 Agent（流式） =====
            if phase == "optimizing":
                print(f"[Orchestrator] 路由 → OptimizeAgent（流式）")
                # 绑定当前会话的 tool_executor
                self.optimize_agent.tool_executor = ctx.get("tool_executor")
                stream = self.optimize_agent.optimize(
                    assessment_context=ctx["assessment_context"],
                    resume_text=ctx["resume_text"],
                    user_message=user_message,
                    conversation_summary=ctx["conversation_summary"],
                    recent_messages=ctx["recent_messages"],
                    memory_context=ctx["memory_context"],
                )
            elif phase == "summary":
                print(f"[Orchestrator] 路由 → ReportAgent（流式）")
                stream = self.report_agent.generate_report(
                    assessment_context=ctx["assessment_context"],
                    resume_text=ctx["resume_text"],
                    conversation_summary=ctx["conversation_summary"],
                    memory_context=ctx["memory_context"],
                    recent_messages=ctx["recent_messages"],
                )
            else:
                # 兜底
                print(f"[Orchestrator] 兜底路由 → OptimizeAgent（阶段: {phase}，流式）")
                self.optimize_agent.tool_executor = ctx.get("tool_executor")
                stream = self.optimize_agent.optimize(
                    assessment_context=ctx["assessment_context"],
                    resume_text=ctx["resume_text"],
                    user_message=user_message,
                    conversation_summary=ctx["conversation_summary"],
                    recent_messages=ctx["recent_messages"],
                    memory_context=ctx["memory_context"],
                )

            # 透传流式输出，同时收集完整回复
            for chunk in stream:
                full_reply += chunk
                yield chunk

            # 保存完整回复
            if full_reply:
                self.session_manager.add_message(session_id, "assistant", full_reply)

            # 后处理
            self._post_chat_processing(session_id)

        except Exception as e:
            print(f"[Orchestrator] 流式对话失败: {e}")
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
