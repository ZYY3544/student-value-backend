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

import re
import uuid
import json
import time
import os
import threading
from datetime import datetime
from typing import Dict, Optional, List, Generator, Tuple
from config import config
from multi_agent import DiagnosisAgent, OptimizeAgent, ReportAgent, PlanningAgent
from tool_executor import ToolExecutor
from utils import safe_json_parse

# Supabase 客户端（延迟导入，可能未安装）
_supabase_client = None
try:
    _sb_url = os.getenv('SUPABASE_URL', '')
    _sb_key = os.getenv('SUPABASE_SERVICE_KEY', '')
    if _sb_url and _sb_key:
        from supabase import create_client as _sb_create
        _supabase_client = _sb_create(_sb_url, _sb_key)
except Exception:
    pass


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

    def create_session(self, assessment_context: dict, resume_text: str,
                       user_id: str = None, assessment_id: str = None) -> str:
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
            "user_id": user_id,  # Supabase 用户 ID（可选）
            "assessment_id": assessment_id,  # Supabase 评估记录 ID（可选）
            "optimization_plan": None,  # PlanningAgent 生成的优化计划
            "pending_phase_transition": None,  # 待确认的阶段转换（如 "summary"）
            "hallucination_warning": None,  # 幻觉检测警告
            "reeval_suggested": False,  # 是否已建议过重评估
            "resume_versions": {},  # 多版本简历 {version_id: {label, resume_text, target_jd, created_at}}
            "jd_auto_suggested": False,  # 是否已主动建议过 JD 定制
        }

        with self._lock:
            self._sessions[session_id] = session

        print(f"[Orchestrator] 创建会话 {session_id[:8]}...")
        self._persist_session_to_supabase(session_id, session)
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
        self._persist_message_to_supabase(session_id, role, content)

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

    def _persist_session_to_supabase(self, session_id: str, session: dict):
        """异步将 chat session 写入 Supabase（非阻塞）"""
        if not _supabase_client:
            return
        def _do():
            try:
                user_id = session.get('user_id')
                assessment_id = session.get('assessment_id')
                if not user_id:
                    return
                _supabase_client.table('chat_sessions').upsert({
                    'id': session_id,
                    'user_id': user_id,
                    'assessment_id': assessment_id,
                    'phase': session.get('phase', 'opening'),
                    'conversation_memory': session.get('memory', ConversationMemory()).to_context_string() if session.get('memory') else None,
                }).execute()
            except Exception as e:
                print(f"[Supabase] chat_sessions 持久化失败: {e}")
        threading.Thread(target=_do, daemon=True).start()

    def save_session_summary(self, session_id: str, summary: str):
        """将会话摘要保存到 Supabase"""
        if not _supabase_client:
            return
        def _do():
            try:
                _supabase_client.table('chat_sessions').update({
                    'summary': summary,
                }).eq('id', session_id).execute()
                print(f"[Supabase] 会话摘要已保存: {session_id[:8]}")
            except Exception as e:
                print(f"[Supabase] 会话摘要保存失败: {e}")
        threading.Thread(target=_do, daemon=True).start()

    def save_resume_version(self, session_id: str, version_id: str,
                            label: str, resume_text: str, target_jd: str = ""):
        """保存简历版本到 session + Supabase"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session["resume_versions"][version_id] = {
                    "label": label,
                    "resume_text": resume_text,
                    "target_jd": target_jd[:500] if target_jd else "",
                    "created_at": time.time(),
                }
        # 异步持久化到 Supabase
        if _supabase_client:
            def _do():
                try:
                    session = self._sessions.get(session_id)
                    user_id = session.get('user_id') if session else None
                    if not user_id:
                        return
                    _supabase_client.table('resume_versions').upsert({
                        'id': f"{session_id}_{version_id}",
                        'session_id': session_id,
                        'user_id': user_id,
                        'version_id': version_id,
                        'label': label,
                        'resume_text': resume_text,
                        'target_jd': target_jd[:500] if target_jd else "",
                    }).execute()
                    print(f"[Supabase] 简历版本已保存: {label}")
                except Exception as e:
                    print(f"[Supabase] 简历版本保存失败: {e}")
            threading.Thread(target=_do, daemon=True).start()

    @staticmethod
    def load_resume_versions(user_id: str) -> list:
        """从 Supabase 加载用户的所有简历版本"""
        if not _supabase_client or not user_id:
            return []
        try:
            resp = _supabase_client.table('resume_versions') \
                .select('session_id, label, target_jd, resume_text') \
                .eq('user_id', user_id) \
                .order('created_at', desc=True) \
                .limit(20) \
                .execute()
            # session_id 作为 version_id 使用
            for item in (resp.data or []):
                if 'session_id' in item and 'version_id' not in item:
                    item['version_id'] = item['session_id']
            return resp.data or []
        except Exception as e:
            print(f"[Supabase] 加载简历版本失败: {e}")
            return []

    @staticmethod
    def load_cross_session_memory(user_id: str) -> str:
        """从 Supabase 加载用户最近 5 次有摘要的会话，拼成跨 session 记忆"""
        if not _supabase_client or not user_id:
            return ""
        try:
            resp = _supabase_client.table('chat_sessions') \
                .select('summary, created_at') \
                .eq('user_id', user_id) \
                .not_.is_('summary', 'null') \
                .order('created_at', desc=True) \
                .limit(5) \
                .execute()
            if not resp.data:
                return ""
            parts = []
            for row in reversed(resp.data):  # 按时间正序
                date_str = row['created_at'][:10] if row.get('created_at') else '未知日期'
                parts.append(f"[{date_str}] {row['summary']}")
            memory_text = "\n\n".join(parts)
            print(f"[Supabase] 加载了 {len(resp.data)} 条跨会话记忆")
            return memory_text
        except Exception as e:
            print(f"[Supabase] 加载跨会话记忆失败: {e}")
            return ""

    def _persist_message_to_supabase(self, session_id: str, role: str, content: str):
        """异步将聊天消息写入 Supabase（非阻塞）"""
        if not _supabase_client:
            return
        def _do():
            try:
                # 检查此 session 是否有对应的 db session（带 user_id）
                session = self._sessions.get(session_id)
                if not session or not session.get('user_id'):
                    return
                _supabase_client.table('chat_messages').insert({
                    'session_id': session_id,
                    'role': role,
                    'content': content,
                }).execute()
            except Exception as e:
                print(f"[Supabase] chat_messages 持久化失败: {e}")
        threading.Thread(target=_do, daemon=True).start()


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
    def compress_history(client, model: str, session: dict) -> str:
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
    def extract_memory(client, model: str, session: dict):
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

            result = safe_json_parse(response.choices[0].message.content)

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
# Reflection 幻觉检测
# ===========================================

class ReflectionChecker:
    """
    幻觉检测器 —— 检查改写内容是否引入了原文中不存在的数据

    对比简历原文和 Agent 改写内容，检测编造的数字/公司/角色/技能。
    后台异步执行，检测到幻觉时存入 session，下一轮对话时警告用户。
    """

    SYSTEM_PROMPT = """你是一个严格的事实核查专家。请对比用户的简历原文和AI改写后的内容，检查改写中是否引入了原文中不存在的信息。

重点检查以下类型的幻觉：
1. 编造的具体数字（如原文没提到的百分比、金额、用户量等）
2. 编造的公司名、项目名、产品名
3. 编造的职位/角色（如把"参与"升级为"主导"、把"实习"说成"全职"）
4. 编造的技能或工具（原文没提到的技术栈）
5. 编造的成果或荣誉

输出严格 JSON 格式：
{
    "has_hallucination": true/false,
    "issues": [
        "具体描述发现的幻觉问题（每条一句话）"
    ]
}

注意：
- 合理的措辞润色不算幻觉（如"负责"改为"主导"在语义合理范围内不算）
- 只标记明确编造的、原文中完全没有依据的信息
- 如果改写使用了「[待补充: ...]」标记，这不算幻觉
- 如果没有发现幻觉，issues 为空数组"""

    @staticmethod
    def check(client, model: str, resume_text: str, rewrite_text: str) -> Optional[dict]:
        """
        检查改写内容是否存在幻觉

        Args:
            client: OpenAI 客户端
            model: 模型名称
            resume_text: 简历原文
            rewrite_text: 改写后的内容

        Returns:
            检测结果 dict，失败返回 None
        """
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ReflectionChecker.SYSTEM_PROMPT},
                    {"role": "user", "content": f"【简历原文】\n{resume_text[:3000]}\n\n【改写内容】\n{rewrite_text[:2000]}"}
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            result = safe_json_parse(response.choices[0].message.content)
            if result.get("has_hallucination"):
                print(f"[ReflectionChecker] 检测到幻觉: {result.get('issues', [])}")
            else:
                print(f"[ReflectionChecker] 未检测到幻觉")
            return result
        except Exception as e:
            print(f"[ReflectionChecker] 检测失败（非阻塞）: {e}")
            return None


# ===========================================
# 简历结构化拆分
# ===========================================

def split_resume_sections(client, model: str, resume_text: str) -> List[dict]:
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
        result = safe_json_parse(content)

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

    def __init__(self, client, model: str = "glm-4.5",
                 llm_service=None, convergence_engine=None,
                 model_router=None):
        self.client = client
        self.model = model
        self.llm_service = llm_service
        self.convergence_engine = convergence_engine
        self.model_router = model_router
        self.session_manager = SessionManager(ttl_seconds=3600)

        # 初始化子 Agent
        _model_plus = model_router.glm_model_plus if model_router else model
        _model_flash = model_router.glm_model_flash if model_router else model
        self.diagnosis_agent = DiagnosisAgent(client, _model_plus)
        self.planning_agent = PlanningAgent(client, _model_plus)
        self.optimize_agent = OptimizeAgent(client, model)
        self.report_agent = ReportAgent(client, model)
        self._model_flash = _model_flash

        # OpenRouter Sonnet 客户端（报告解读专用）
        self.sonnet_client = None
        self.sonnet_model = config.SONNET_MODEL_ID
        if config.OPENROUTER_API_KEY:
            try:
                from openai import OpenAI
                self.sonnet_client = OpenAI(
                    api_key=config.OPENROUTER_API_KEY,
                    base_url=config.OPENROUTER_BASE_URL,
                )
                print(f"[Orchestrator] OpenRouter Sonnet 初始化成功: {self.sonnet_model}")
            except Exception as e:
                print(f"[Orchestrator] OpenRouter 初始化失败，报告解读将回退到主力模型: {e}")
        else:
            print("[Orchestrator] OPENROUTER_API_KEY 未配置，报告解读将使用主力模型")

        haiku_id = model_router.haiku_model if model_router else "N/A"
        print("[Orchestrator] 多 Agent 系统初始化完成")
        print(f"  - 报告解读 (Sonnet):  {self.sonnet_model}")
        print(f"  - 主力模型:           {haiku_id}")
        print(f"  - DiagnosisAgent: 开场诊断 → {_model_plus}（温度 {DiagnosisAgent.TEMPERATURE}）")
        print(f"  - PlanningAgent:  优化规划 → {_model_plus}（温度 {PlanningAgent.TEMPERATURE}）")
        print(f"  - OptimizeAgent:  简历优化 → {model}（温度 {OptimizeAgent.TEMPERATURE}）")
        print(f"  - ReportAgent:    总结报告 → {model}（温度 {ReportAgent.TEMPERATURE}）")
        print(f"  - 简历拆分:       → {_model_flash}")
        if model_router:
            print(f"  - ModelRouter:    Haiku/GLM 自动切换已启用")
        if llm_service and convergence_engine:
            print(f"  - ToolExecutor:   Function Call 工具调用已启用")

    def _resolve_model_for_user(self, user_id: str = None):
        """
        根据用户 ID 解析应使用的模型，并更新所有子 Agent

        如果配置了 model_router，则根据用户用量选择 Haiku 或 GLM；
        否则回退到默认的 client/model。

        Returns:
            (client, model, provider) 三元组
        """
        if not self.model_router:
            return self.client, self.model, "glm"

        try:
            client, model, provider = self.model_router.get_client_for_user(user_id)
        except RuntimeError:
            # 所有模型不可用，回退默认
            print("[Orchestrator] ModelRouter 无可用模型，回退默认 client")
            return self.client, self.model, "glm"

        # 更新所有子 Agent 的 client、model
        for agent in [self.diagnosis_agent, self.planning_agent,
                      self.optimize_agent, self.report_agent]:
            agent.client = client
            agent.model = model

        return client, model, provider

    def start_session(self, assessment_context: dict, resume_text: str,
                      user_id: str = None, assessment_id: str = None,
                      resume_sections: list = None,
                      preloaded_greeting: str = None) -> dict:
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
            user_id: Supabase 用户 ID（可选，用于数据持久化）
            assessment_id: Supabase 评估记录 ID（可选）

        Returns:
            {"session_id": "...", "greeting": "..."}
        """
        import time as _time
        _t0 = _time.time()
        def _lap(label):
            nonlocal _t0
            now = _time.time()
            print(f"[start_session] {label}: {now - _t0:.2f}s")
            _t0 = now

        # 根据用户解析模型（GLM/Sonnet）
        active_client, active_model, active_provider = self._resolve_model_for_user(user_id)
        _lap("resolve_model")

        session_id = self.session_manager.create_session(
            assessment_context=assessment_context,
            resume_text=resume_text,
            user_id=user_id,
            assessment_id=assessment_id,
        )

        _lap("create_session")
        # 记录当前会话使用的模型提供商
        self.session_manager.update_session(session_id, {"active_provider": active_provider})

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
            # 注入版本持久化回调
            _sid = session_id
            _sm = self.session_manager
            tool_executor._on_version_saved = lambda vid, label, text, jd: \
                _sm.save_resume_version(_sid, vid, label, text, jd)
            # 加载用户之前保存的简历版本
            if user_id:
                saved_versions = SessionManager.load_resume_versions(user_id)
                for v in saved_versions:
                    vid = v.get('version_id')
                    if not vid:
                        continue
                    tool_executor._resume_versions[vid] = {
                        "label": v.get('label', ''),
                        "resume_text": v.get('resume_text', ''),
                        "target_jd": v.get('target_jd', ''),
                        "created_at": 0,
                    }
                if saved_versions:
                    print(f"[Orchestrator] 已加载 {len(saved_versions)} 个历史简历版本")
            # 存到 session 中，后续 chat/chat_stream 可取用
            self.session_manager.update_session(session_id, {
                "tool_executor": tool_executor,
            })

        _lap("tool_executor_setup")
        # 加载跨 session 记忆（让 Agent "记住"之前的对话）
        cross_session_memory = SessionManager.load_cross_session_memory(user_id)
        if cross_session_memory:
            self.session_manager.update_session(session_id, {
                "cross_session_memory": cross_session_memory,
            })
            print(f"[Orchestrator] 已注入跨会话记忆")

        _lap("load_cross_session_memory")
        # ===== 个性化开场白 =====
        if preloaded_greeting:
            greeting = preloaded_greeting
            print(f"[Orchestrator] 使用评估阶段预生成的开场白（{len(greeting)}字）")
        else:
            print(f"[Orchestrator] 调用 DiagnosisAgent 生成个性化开场白")
            greeting = self.diagnosis_agent.diagnose(assessment_context, resume_text)

        _lap("greeting")
        # 后台任务共用的 client/model（提前绑定，避免分支遗漏）
        _bg_client, _bg_model = active_client, active_model

        # 简历结构拆分：如果评测阶段已预拆分，直接使用；否则后台异步拆分（不阻塞 start 响应）
        if resume_sections and isinstance(resume_sections, list) and len(resume_sections) > 0:
            self.session_manager.update_session(session_id, {
                "resume_sections": resume_sections
            })
            print(f"[Orchestrator] 使用预拆分简历段落（{len(resume_sections)} 段），跳过 LLM 拆分")
        else:
            # 后台异步拆分，不阻塞 /chat/start 响应
            _split_client, _split_model = _bg_client, self._model_flash
            _split_session_id = session_id
            _split_sm = self.session_manager
            def _bg_split():
                try:
                    sections = split_resume_sections(_split_client, _split_model, resume_text)
                    _split_sm.update_session(_split_session_id, {
                        "resume_sections": sections
                    })
                    print(f"[Orchestrator] 后台简历拆分完成（{len(sections)} 段）")
                except Exception as e:
                    print(f"[Orchestrator] 后台简历拆分失败，使用整篇兜底: {e}")
                    _split_sm.update_session(_split_session_id, {
                        "resume_sections": [{"type": "other", "title": "完整简历", "content": resume_text[:3000]}]
                    })
            threading.Thread(target=_bg_split, daemon=True).start()
            # 先给一个临时的整篇兜底，后台线程完成后会覆盖
            resume_sections = [{"type": "other", "title": "完整简历", "content": resume_text[:3000]}]
            print(f"[Orchestrator] 简历拆分已移至后台，先返回临时整篇段落")

        # PlanningAgent 优化计划已移除
        # （该计划仅在用户主动聊天时作为 OptimizeAgent 的辅助上下文，
        #   但多数用户不会聊天，白耗一次 Haiku 调用）

        # 保存到对话历史
        self.session_manager.add_message(session_id, "user", "你好，帮我看看简历怎么改")
        self.session_manager.add_message(session_id, "assistant", greeting)

        # 开场完成后进入优化阶段
        self.session_manager.update_session(session_id, {"phase": "optimizing"})

        _lap("save_messages_and_finalize")
        return {
            "session_id": session_id,
            "greeting": greeting,
            "resume_sections": resume_sections,
        }

    def _get_agent_context(self, session: dict) -> dict:
        """
        从 session 中提取子 Agent 需要的上下文

        这是协调者的核心工作：把正确的上下文传给正确的 Agent
        """
        memory: ConversationMemory = session["memory"]
        messages = session["messages"]
        compressed = session.get("compressed_history", "")

        # 构建对话摘要：跨 session 记忆 + 压缩历史
        cross_memory = session.get("cross_session_memory", "")
        conversation_summary = ""
        if cross_memory:
            conversation_summary = f"【历史对话记忆（之前的会话）】\n{cross_memory}\n\n"
        if compressed:
            conversation_summary += compressed

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
            "optimization_plan": session.get("optimization_plan"),
        }

    # 用户确认肯定词
    CONFIRM_YES = ["好的", "好", "是的", "是", "对", "嗯", "可以", "行", "OK", "ok", "没问题", "总结吧", "做个总结"]
    # 用户否定词
    CONFIRM_NO = ["不", "不要", "不用", "继续", "还没", "先不", "再改改", "等等", "算了"]

    def _detect_phase_transition(self, user_message: str, current_phase: str,
                                  session: dict = None) -> tuple:
        """
        检测阶段转换（支持确认机制）

        规则：
        1. 如果有 pending_phase_transition="summary" → 检查用户是肯定还是否定
        2. optimizing → 首次命中总结关键词 → 返回确认提示，不直接切
        3. summary → optimizing：用户在总结后又想继续优化（回退）

        Returns:
            (new_phase, confirmation_prompt_or_None)
        """
        msg = user_message.strip()

        # ===== 处理待确认的阶段转换 =====
        pending = session.get("pending_phase_transition") if session else None
        if pending == "summary":
            # 检查用户回复是肯定还是否定
            if any(kw in msg for kw in self.CONFIRM_YES):
                return ("summary", None)
            elif any(kw in msg for kw in self.CONFIRM_NO):
                return ("optimizing", None)
            # 既不肯定也不否定，当作否定（继续优化）
            return ("optimizing", None)

        # ===== 正常阶段转换检测 =====
        has_continue = any(kw in msg for kw in self.CONTINUE_KEYWORDS)
        has_summary = any(kw in msg for kw in self.SUMMARY_KEYWORDS)

        if current_phase == "optimizing":
            # 有总结意图且没有继续意图 → 先确认
            if has_summary and not has_continue:
                # 额外检查：消息太长（>30字）可能只是聊天中偶然提到关键词
                if len(msg) > 30 and not msg.endswith(("吧", "了", "。")):
                    return (current_phase, None)
                # 返回确认提示，不直接切换
                return (current_phase, "看起来你准备结束优化了，需要我帮你做个总结吗？")

        elif current_phase == "summary":
            # 总结阶段后用户想继续优化 → 回退
            if has_continue:
                print(f"[Orchestrator] 用户在总结后继续优化，回退到 optimizing")
                return ("optimizing", None)

        return (current_phase, None)

    def _post_chat_processing(self, session_id: str):
        """
        每轮对话后的后处理（后台线程执行，不阻塞用户）

        1. 提取结构化记忆（供下一轮子 Agent 使用）
        2. 判断是否需要压缩对话历史
        """
        # 捕获当前 client/model 到闭包（线程安全）
        _pc_client = self.optimize_agent.client
        _pc_model = self.optimize_agent.model

        def _bg_process():
            session = self.session_manager.get_session(session_id)
            if not session:
                return

            # 1. 提取结构化记忆
            try:
                HistoryCompressor.extract_memory(_pc_client, _pc_model, session)
            except Exception as e:
                print(f"[Orchestrator] 记忆提取失败: {e}")

            # 2. 判断是否需要压缩
            if HistoryCompressor.should_compress(session):
                try:
                    compressed = HistoryCompressor.compress_history(
                        _pc_client, _pc_model, session
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

            # 3. 幻觉检测（仅在 Agent 做了简历段落改写时触发）
            try:
                messages = session["messages"]
                if messages and messages[-1].get("role") == "assistant":
                    last_reply = messages[-1]["content"]

                    # 改写标记：必须有明确的改写输出格式
                    rewrite_markers = ["EDIT>>>", "建议改为", "修改后", "润色后"]
                    has_rewrite = any(marker in last_reply for marker in rewrite_markers)

                    # 排除非改写场景（搜索分析、岗位对比等不应触发）
                    skip_markers = ["搜索", "联网查询", "岗位信息", "匹配度", "搜一下", "正在获取"]
                    is_search_reply = any(marker in last_reply for marker in skip_markers)

                    if has_rewrite and not is_search_reply:
                        resume_text = session.get("resume_text", "")
                        result = ReflectionChecker.check(
                            _pc_client, _pc_model, resume_text, last_reply
                        )
                        if result and result.get("has_hallucination") and result.get("issues"):
                            self.session_manager.update_session(session_id, {
                                "hallucination_warning": result["issues"]
                            })
            except Exception as e:
                print(f"[Orchestrator] 幻觉检测失败（非阻塞）: {e}")

            # 4. 生成跨 session 摘要（每 6 条消息更新一次）
            msg_count = session.get("message_count", 0)
            if msg_count >= 4 and msg_count % 6 == 0:
                try:
                    messages = session["messages"]
                    conv_text = "\n".join(
                        f"{'用户' if m['role']=='user' else '顾问'}: {m['content'][:200]}"
                        for m in messages[-10:]
                    )
                    summary_prompt = """请用2-3句话概括以下对话的要点，包括：用户关心什么、给了什么建议、达成了什么共识。
要求简洁，100字以内。"""
                    resp = _pc_client.chat.completions.create(
                        model=_pc_model,
                        messages=[
                            {"role": "system", "content": summary_prompt},
                            {"role": "user", "content": conv_text},
                        ],
                        temperature=0.0,
                    )
                    summary = resp.choices[0].message.content.strip()
                    self.session_manager.save_session_summary(session_id, summary)
                except Exception as e:
                    print(f"[Orchestrator] 会话摘要生成失败: {e}")

        threading.Thread(target=_bg_process, daemon=True).start()

    @staticmethod
    def _parse_action(user_message: str) -> Tuple[Optional[str], str]:
        """解析消息中的 [ACTION:xxx] 前缀，返回 (action, actual_message)"""
        m = re.match(r'^\[ACTION:(.+?)\]\s*(.*)', user_message, re.DOTALL)
        if m:
            return m.group(1), m.group(2).strip() or '开始吧'
        return None, user_message

    def _check_reeval_suggestion(self, session: dict) -> Optional[str]:
        """
        检查是否应该建议用户重评估

        条件：已优化 >= 2 个段落 且 尚未建议过
        Returns:
            建议提示文本，或 None
        """
        if session.get("reeval_suggested"):
            return None
        memory: ConversationMemory = session["memory"]
        if len(memory.optimized_sections) >= 2:
            return (
                "你已经优化了 {n} 个段落（{sections}），要不要我帮你重新跑一遍评估，"
                "看看能力评分和薪酬定位有什么变化？"
            ).format(
                n=len(memory.optimized_sections),
                sections="、".join(memory.optimized_sections[:3]),
            )
        return None

    def _check_jd_auto_suggestion(self, session: dict) -> Optional[str]:
        """
        代码级自动触发：改完简历后主动建议 JD 定制

        条件：已优化 >= 1 个段落 且 尚未建议过 JD 定制 且 用户有明确目标岗位
        """
        if session.get("jd_auto_suggested"):
            return None
        memory: ConversationMemory = session["memory"]
        ctx = session.get("assessment_context", {})
        job_title = ctx.get("jobTitle", "")

        # 不再使用固定话术，由 LLM 在对话中自然引导
        return None

    def _build_report_analysis_prompt(self, session: dict) -> str:
        """构建深度报告解读的 System Prompt（V3: 配合新版评测报告结构）"""
        ctx = session["assessment_context"]
        resume_text = session["resume_text"]

        abilities = ctx.get("abilities", {})
        text_preview = resume_text[:2000]
        salary_range = ctx.get("salaryRange", "未知")
        city = ctx.get("city", "未知")
        job_title = ctx.get("jobTitle", "未知")

        # 构建5维能力得分文本
        abilities_lines = []
        for name, info in abilities.items():
            score_10 = info.get("score", 50) / 10
            abilities_lines.append(f"  - {name}：{score_10:.1f}分")
        abilities_text = "\n".join(abilities_lines)

        # 能力层级
        level_title = ctx.get("levelTitle", "未知")
        level_description = ctx.get("levelDescription", "")

        # 简历表达力数据
        expression_score = ctx.get("expressionScore", "未知")
        star_score = ctx.get("starScore", "未知")
        keyword_score = ctx.get("keywordScore", "未知")
        quantify_score = ctx.get("quantifyScore", "未知")
        completeness_score = ctx.get("completenessScore", "未知")
        structure_score = ctx.get("structureScore", "未知")
        power_score = ctx.get("powerScore", "未知")

        # 多方向薪酬对比数据
        job_comparisons = ctx.get("jobComparisons", [])
        if job_comparisons:
            job_comp_lines = []
            for jc in job_comparisons:
                fn = jc.get("jobFunction", "未知")
                sr = jc.get("salaryRange", "未知")
                ms = jc.get("matchScore", "未知")
                job_comp_lines.append(f"  - {fn}：{sr}（匹配度 {ms}%）")
            job_comp_text = "\n".join(job_comp_lines)
        else:
            job_comp_text = f"  - {ctx.get('jobFunction', job_title)}：{salary_range}"

        return f"""你是 Sparky，一个专业的求职分析 agent。你的风格是：精准、有判断力、不废话。你不讨好用户，也不说教，而是用数据说话，帮用户看清事实。

## 你的任务
基于用户的评测报告数据，生成一段个性化的深度解读。核心目标是帮用户理解三件事：
1. 我的能力底子在什么位置
2. 这个底子在不同方向上值多少钱
3. 我的简历有没有把底子讲清楚

## 最重要的前置规则：输入质量判断
如果简历内容是乱码、测试文字或过于简短，直接告知用户内容不足，不要编造分析。

## 评估数据
- 城市：{city} | 目标方向：{ctx.get('jobFunction', job_title)} | 目标职位：{job_title}
- 学历：{ctx.get('educationLevel', '未知')} | 专业：{ctx.get('major', '未知')}
- 意向行业：{ctx.get('industry', '未知')} | 企业性质：{ctx.get('companyType', '未知')}
- 能力层级：{level_title}（{level_description}）
- 5维能力画像（满分10分）：
{abilities_text}
- 多方向薪酬对比（第一个为用户的目标方向）：
{job_comp_text}
- 简历表达力综合分：{expression_score}/100
  STAR规范度 {star_score} | 关键词覆盖 {keyword_score} | 量化程度 {quantify_score}
  表达力度 {power_score} | 信息完整度 {completeness_score} | 结构规范度 {structure_score}

## 简历原文
{text_preview}

## 思考步骤（内部推理，不要输出）
1. 看五维度的整体形状——是均衡型还是有明显的高低差？这种形状对目标岗位意味着什么？
2. 三个方向的薪酬差异有多大？差异背后的原因是什么（赛道本身的薪酬水位 vs 能力匹配度）？
3. 简历表达力的六个维度里，哪个最低？这个低分对目标岗位的影响大不大？
4. 简历里有哪些经历跟目标岗位有关联但没有被充分表达？

## 输出格式：严格4段，段间空行分隔

**第一段：开头（2-3句）**
- 第一句直接进入："我帮你从三个层面拆解一下这份报告：你的能力底子在什么位置、三个方向分别值多少钱、以及简历有没有把你的真实水平表达出来。"
- 第二句用该用户的实际数据引出核心认知——同一份简历在不同方向上估值差异很大。直接引用具体的薪酬数字做对比。
- 不要说教（"先说一个很多同学不知道的事"），不要宣布（"你的评测结果出来了"），直接用数据引出发现。

**第二段：你的能力底子（3-4句）**
- 点明能力层级称谓，用一句话说清楚这意味着什么能力状态
- 看五维度的整体形状，给出一个有判断力的概括（不要逐个念分数）
- 把能力形状跟目标方向关联起来——这种形状对投{ctx.get('jobFunction', job_title)}意味着什么
- 段落标题用 **你的能力底子** 加粗

**第三段：市场怎么给你定价（3-4句）**
- 严格使用上方"多方向薪酬对比"中列出的数据，禁止编造或推测薪酬数字
- 列出各方向的薪酬估值做对比，注意第一个方向是用户的目标方向
- 解释薪酬差异的原因——是赛道本身的薪酬水位不同，还是能力匹配度不同，还是两者都有
- 结合简历中的具体经历，判断哪个方向最值得优先投入
- 段落标题用 **市场怎么给你定价** 加粗

**第四段：简历有没有把底子讲清楚（3-4句）**
- 点明简历表达力综合分，一句话概括整体情况
- 指出最弱的1-2个维度，用具体数据说明
- 说清楚改善后能带来什么变化（表达力提升 → 薪酬估值跟着走）
- 结尾引导用户进入简历优化："这也是改写简历能给你带来最大回报的地方"
- 段落标题用 **简历有没有把底子讲清楚** 加粗

## 语气要求
- 像一个很强的顾问在做情况评估，不是老师在讲课，不是学长在聊天
- 有判断力：敢给方向性建议（"XX方向值得优先投入"），不要说"需要你自己权衡"
- 用数据说话：每个判断都要有具体数字支撑，薪酬数字必须严格引用评估数据中的原始数值
- 正面框架为主：说"你的经历跟目标方向有不少连接点"，不说"你的简历有问题"
- 负面信息用中性语气过渡："坦白说""不过""目前"
- 不要空洞的赞美（"你的经历很亮眼"），但可以基于事实给正面判断

## 语气示范

好的开头：
"我帮你从三个层面拆解一下这份报告：你的能力底子在什么位置、三个方向分别值多少钱、以及简历有没有把你的真实水平表达出来。先说一个你对比三个方向就能发现的事——同一份简历，产品管理的估值是14.3k-16.7k，数据分析只有11.9k-14.0k，差了将近3k。"

差的开头（说教型）：
"先说一个很多同学不知道的事：你的市场价值不是一个固定数字。同一个人，选不同的赛道、用不同的方式写简历，市场给出的估值可以差很多。"

差的开头（宣布型）：
"你的评测结果出来了，我帮你从三个层面拆解一下。"

## 硬性约束
- 全文 400-550 字
- 4段结构，段间空行分隔
- 段落标题用 **加粗**
- 禁止 emoji
- 禁止提及"职级""等级""level"、具体分数（五维度的分不要报数字，用"相对高/低"来描述）
- 禁止提及评测方法论术语（HAY、八因素等）
- 禁止说教语气（"先说一个很多同学不知道的事""你需要知道的是"）
- 禁止空洞赞美
- 关键能力词用 **加粗** 高亮
- 输出到第四段结束，不要输出引导问题（引导问题由系统自动追加）"""

    def _call_fallback_stream(self, system_prompt: str) -> Generator[str, None, None]:
        """备用模型流式调用（供报告解读 fallback 使用）"""
        stream = self.diagnosis_agent.client.chat.completions.create(
            model=self.diagnosis_agent.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "请基于以上信息，生成深度洞察分析。"}
            ],
            temperature=0.5,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _stream_report_analysis(self, session: dict) -> Generator[str, None, None]:
        """报告解读流式输出（AB测试：临时切换为 Haiku 主力模型），引导问题由代码追加"""
        system_prompt = self._build_report_analysis_prompt(session)

        try:
            # AB测试：暂时跳过 Sonnet，直接用 Haiku 主力模型
            print(f"[Orchestrator] 报告解读使用主力模型（AB测试）: {self.diagnosis_agent.model}")
            yield from self._call_fallback_stream(system_prompt)

            # 程序化追加引导问题
            ctx = session["assessment_context"]
            abilities = ctx.get("abilities", {})
            job_function = ctx.get("jobFunction", ctx.get("jobTitle", "未知"))
            city = ctx.get("city", "未知")
            # 取目标方向的薪酬
            jc_list = ctx.get("jobComparisons", [])
            target_salary = jc_list[0].get("salaryRange", "未知") if jc_list else ctx.get("salaryRange", "未知")

            sorted_abs = sorted(abilities.items(), key=lambda x: x[1].get("score", 50))
            weakest = sorted_abs[0][0] if sorted_abs else "某项能力"

            guidance = f"\n\n---\n\n**你可能还想了解：**\n"
            guidance += f"1. 我的5维能力画像是怎么生成的？是简历里哪些内容决定了这些分数？\n"
            guidance += f"2. {target_salary}这个估值区间的数据来源是什么？为什么和我了解到的{city}{job_function}校招薪资有差异？\n"
            guidance += f"3. 「{weakest}」这个维度具体在衡量什么能力？为什么我在这项上得分相对较低？"

            yield guidance

        except Exception as e:
            import traceback
            print(f"[Orchestrator] 报告解读生成失败: {type(e).__name__}: {e}")
            traceback.print_exc()
            yield f"抱歉，生成报告解读时遇到了问题（{type(e).__name__}），请再试一次。"

    # ===== 分层引导问题系统 =====

    def _get_guidance_layer(self, session: dict) -> int:
        """根据报告解读后的对话轮数，计算当前引导层级（0=未激活, 1/2/3=三层）"""
        if not session.get("report_analysis_done"):
            return 0
        rounds = session.get("post_report_rounds", 0)
        if rounds <= 2:
            return 2  # Layer 1 已在报告解读中输出，后续从 Layer 2 开始
        elif rounds <= 5:
            return 3
        else:
            return 0  # 超过一定轮数后停止引导，让用户自由对话

    def _build_guidance_questions(self, session: dict) -> str:
        """根据引导层级，用评估数据生成个性化引导问题"""
        layer = self._get_guidance_layer(session)
        if layer <= 0:
            return ""

        ctx = session.get("assessment_context", {})
        abilities = ctx.get("abilities", {})
        salary_range = ctx.get("salaryRange", "未知")
        job_title = ctx.get("jobTitle", "未知")

        # 按得分排序找出最强/最弱维度
        sorted_abs = sorted(abilities.items(), key=lambda x: x[1].get("score", 50))
        weakest_name = sorted_abs[0][0] if sorted_abs else "某项能力"
        strongest_name = sorted_abs[-1][0] if len(sorted_abs) > 1 else "某项能力"

        if layer == 2:
            # 第二层：个人洞察 — 报告对我意味着什么
            questions = [
                f"报告里说我的「{weakest_name}」相对偏弱，在校期间有什么具体方法能快速补上来？",
                f"我的「{strongest_name}」是亮点，面试{job_title}时怎么把这个优势讲出来？",
                f"对于{job_title}这个岗位，面试官最看重简历里的哪些经历？",
            ]
        elif layer == 3:
            # 第三层：行动落地 — 推动学生去改简历、做规划
            questions = [
                f"能帮我看看简历整体怎么调整，让它更匹配{job_title}吗？",
                f"除了{job_title}，根据我的背景还有哪些岗位方向值得考虑？",
                f"如果要在3个月内提升自己的岗位匹配度，你建议我优先做哪几件事？",
            ]
        else:
            return ""

        lines = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        return f"\n\n---\n\n💡 **你可能还想了解：**\n{lines}"

    def chat(self, session_id: str, user_message: str, canvas_mode: bool = False) -> Optional[str]:
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
            canvas_mode: 是否处于画布模式

        Returns:
            Agent 的回复文本，会话无效返回 None
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return None

        # 根据用户解析模型
        user_id = session.get("user_id")
        self._resolve_model_for_user(user_id)

        # 解析 [ACTION:xxx] 前缀
        action, actual_message = self._parse_action(user_message)

        # 保存用户消息
        self.session_manager.add_message(session_id, "user", actual_message)

        # 阶段转换检测（带确认机制）
        new_phase, confirmation_prompt = self._detect_phase_transition(
            actual_message, session["phase"], session
        )

        # 清除 pending（无论结果如何）
        if session.get("pending_phase_transition"):
            self.session_manager.update_session(session_id, {"pending_phase_transition": None})

        if confirmation_prompt:
            # 需要用户确认 → 设置 pending，直接返回确认提示
            self.session_manager.update_session(session_id, {"pending_phase_transition": "summary"})
            self.session_manager.add_message(session_id, "assistant", confirmation_prompt)
            return confirmation_prompt

        if new_phase != session["phase"]:
            print(f"[Orchestrator] 阶段转换: {session['phase']} → {new_phase}")
            self.session_manager.update_session(session_id, {"phase": new_phase})

        # 重新获取更新后的 session
        session = self.session_manager.get_session(session_id)
        phase = session["phase"]

        # 提取子 Agent 需要的上下文
        ctx = self._get_agent_context(session)

        # ===== 幻觉警告检查 =====
        hallucination_warning = session.get("hallucination_warning")
        warning_prefix = ""
        if hallucination_warning:
            issues_text = "\n".join(f"  - {issue}" for issue in hallucination_warning)
            warning_prefix = f"提醒：上次改写中可能包含原文中没有的信息，请注意核实：\n{issues_text}\n\n"
            self.session_manager.update_session(session_id, {"hallucination_warning": None})

        # ===== 路由 =====
        try:
            if action == "解读报告":
                print(f"[Orchestrator] 路由 → 深度报告解读（非流式）")
                reply = "".join(self._stream_report_analysis(session))
                # 标记报告解读已完成，启动分层引导
                self.session_manager.update_session(session_id, {
                    "report_analysis_done": True,
                    "post_report_rounds": 0,
                })

            elif action in ("润色项目经历", "模拟面试"):
                enhanced_message = f"用户选择了「{action}」功能。用户补充说：{actual_message}。请直接开始执行「{action}」。"
                self.optimize_agent.tool_executor = ctx.get("tool_executor")
                reply = self.optimize_agent.optimize_sync(
                    assessment_context=ctx["assessment_context"],
                    resume_text=ctx["resume_text"],
                    user_message=enhanced_message,
                    conversation_summary=ctx["conversation_summary"],
                    recent_messages=ctx["recent_messages"],
                    memory_context=ctx["memory_context"],
                    canvas_mode=canvas_mode,
                    optimization_plan=ctx.get("optimization_plan"),
                )

            elif phase == "optimizing":
                print(f"[Orchestrator] 路由 → OptimizeAgent")
                self.optimize_agent.tool_executor = ctx.get("tool_executor")
                reply = self.optimize_agent.optimize_sync(
                    assessment_context=ctx["assessment_context"],
                    resume_text=ctx["resume_text"],
                    user_message=actual_message,
                    conversation_summary=ctx["conversation_summary"],
                    recent_messages=ctx["recent_messages"],
                    memory_context=ctx["memory_context"],
                    canvas_mode=canvas_mode,
                    optimization_plan=ctx.get("optimization_plan"),
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
                print(f"[Orchestrator] 兜底路由 → OptimizeAgent（阶段: {phase}）")
                self.optimize_agent.tool_executor = ctx.get("tool_executor")
                reply = self.optimize_agent.optimize_sync(
                    assessment_context=ctx["assessment_context"],
                    resume_text=ctx["resume_text"],
                    user_message=actual_message,
                    conversation_summary=ctx["conversation_summary"],
                    recent_messages=ctx["recent_messages"],
                    memory_context=ctx["memory_context"],
                    canvas_mode=canvas_mode,
                    optimization_plan=ctx.get("optimization_plan"),
                )

            # 如果有幻觉警告前缀，拼在回复前面
            if warning_prefix:
                reply = warning_prefix + reply

            # 程序化重评估建议（改完 N 段后自动追加）
            reeval_hint = self._check_reeval_suggestion(session)
            if reeval_hint and phase == "optimizing":
                reply = reply + "\n\n---\n\n" + reeval_hint
                self.session_manager.update_session(session_id, {"reeval_suggested": True})

            # 程序化 JD 定制建议（改完 1 段后主动提示）
            jd_hint = self._check_jd_auto_suggestion(session)
            if jd_hint and phase == "optimizing" and not reeval_hint:
                reply = reply + "\n\n---\n\n" + jd_hint
                self.session_manager.update_session(session_id, {"jd_auto_suggested": True})

            # 分层引导问题（报告解读后的后续对话）
            if action != "解读报告":  # 解读报告本身已含 Layer 1 引导
                guidance = self._build_guidance_questions(session)
                if guidance:
                    reply = reply + guidance
                    # 递增轮次计数
                    rounds = session.get("post_report_rounds", 0)
                    self.session_manager.update_session(session_id, {
                        "post_report_rounds": rounds + 1,
                    })

            # 保存回复
            self.session_manager.add_message(session_id, "assistant", reply)

            # 后处理
            self._post_chat_processing(session_id)

            return reply

        except Exception as e:
            print(f"[Orchestrator] 对话失败: {e}")
            return "抱歉，处理你的消息时遇到了问题，请再试一次。"

    def chat_stream(self, session_id: str, user_message: str, canvas_mode: bool = False) -> Generator[str, None, None]:
        """
        处理用户消息，返回流式回复（SSE 格式）

        协调者流程：
        1. 解析 [ACTION:xxx] 前缀
        2. 保存用户消息
        3. 检测阶段转换
        4. 路由到对应子 Agent（流式）
        5. 收集完整回复并保存
        6. 后处理

        Yields:
            逐块的文本内容
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            yield "[ERROR] 会话已过期或不存在"
            return

        # 根据用户解析模型
        user_id = session.get("user_id")
        self._resolve_model_for_user(user_id)

        # 解析 [ACTION:xxx] 前缀
        action, actual_message = self._parse_action(user_message)
        if action:
            print(f"[Orchestrator] 检测到动作前缀: ACTION={action}, 用户说: {actual_message}")

        # 保存用户消息（存实际文本，不存前缀）
        self.session_manager.add_message(session_id, "user", actual_message)

        # 阶段转换检测（带确认机制）
        new_phase, confirmation_prompt = self._detect_phase_transition(
            actual_message, session["phase"], session
        )

        # 清除 pending（无论结果如何）
        if session.get("pending_phase_transition"):
            self.session_manager.update_session(session_id, {"pending_phase_transition": None})

        if confirmation_prompt:
            # 需要用户确认 → 设置 pending，直接返回确认提示
            self.session_manager.update_session(session_id, {"pending_phase_transition": "summary"})
            self.session_manager.add_message(session_id, "assistant", confirmation_prompt)
            yield confirmation_prompt
            return

        if new_phase != session["phase"]:
            print(f"[Orchestrator] 阶段转换: {session['phase']} → {new_phase}")
            self.session_manager.update_session(session_id, {"phase": new_phase})

        # 重新获取
        session = self.session_manager.get_session(session_id)
        phase = session["phase"]

        # 提取子 Agent 需要的上下文
        ctx = self._get_agent_context(session)

        # ===== 幻觉警告检查 =====
        hallucination_warning = session.get("hallucination_warning")
        if hallucination_warning:
            issues_text = "\n".join(f"  - {issue}" for issue in hallucination_warning)
            warning_text = f"提醒：上次改写中可能包含原文中没有的信息，请注意核实：\n{issues_text}\n\n"
            yield warning_text
            self.session_manager.update_session(session_id, {"hallucination_warning": None})

        full_reply = ""

        try:
            # ===== 特殊动作路由 =====
            if action == "解读报告":
                print(f"[Orchestrator] 路由 → 深度报告解读（专用 Prompt）")
                stream = self._stream_report_analysis(session)
                # 标记报告解读已完成，启动分层引导
                self.session_manager.update_session(session_id, {
                    "report_analysis_done": True,
                    "post_report_rounds": 0,
                })

            elif action in ("润色简历", "润色项目经历", "模拟面试", "职业规划"):
                print(f"[Orchestrator] 路由 → OptimizeAgent（动作: {action}）")
                enhanced_message = f"用户选择了「{action}」功能。用户补充说：{actual_message}。请直接开始执行「{action}」。"
                self.optimize_agent.tool_executor = ctx.get("tool_executor")
                stream = self.optimize_agent.optimize(
                    assessment_context=ctx["assessment_context"],
                    resume_text=ctx["resume_text"],
                    user_message=enhanced_message,
                    conversation_summary=ctx["conversation_summary"],
                    recent_messages=ctx["recent_messages"],
                    memory_context=ctx["memory_context"],
                    canvas_mode=canvas_mode,
                    optimization_plan=ctx.get("optimization_plan"),
                )

            # ===== 常规阶段路由 =====
            elif phase == "optimizing":
                print(f"[Orchestrator] 路由 → OptimizeAgent（流式）")
                self.optimize_agent.tool_executor = ctx.get("tool_executor")
                stream = self.optimize_agent.optimize(
                    assessment_context=ctx["assessment_context"],
                    resume_text=ctx["resume_text"],
                    user_message=actual_message,
                    conversation_summary=ctx["conversation_summary"],
                    recent_messages=ctx["recent_messages"],
                    memory_context=ctx["memory_context"],
                    canvas_mode=canvas_mode,
                    optimization_plan=ctx.get("optimization_plan"),
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
                print(f"[Orchestrator] 兜底路由 → OptimizeAgent（阶段: {phase}，流式）")
                self.optimize_agent.tool_executor = ctx.get("tool_executor")
                stream = self.optimize_agent.optimize(
                    assessment_context=ctx["assessment_context"],
                    resume_text=ctx["resume_text"],
                    user_message=actual_message,
                    conversation_summary=ctx["conversation_summary"],
                    recent_messages=ctx["recent_messages"],
                    memory_context=ctx["memory_context"],
                    canvas_mode=canvas_mode,
                    optimization_plan=ctx.get("optimization_plan"),
                )

            # 透传流式输出，同时收集完整回复
            for chunk in stream:
                full_reply += chunk
                yield chunk

            # 程序化重评估建议（改完 N 段后自动追加）
            reeval_hint = self._check_reeval_suggestion(session)
            if reeval_hint and phase == "optimizing":
                reeval_block = "\n\n---\n\n" + reeval_hint
                full_reply += reeval_block
                yield reeval_block
                self.session_manager.update_session(session_id, {"reeval_suggested": True})

            # 程序化 JD 定制建议（改完 1 段后主动提示）
            jd_hint = self._check_jd_auto_suggestion(session)
            if jd_hint and phase == "optimizing" and not reeval_hint:
                jd_block = "\n\n---\n\n" + jd_hint
                full_reply += jd_block
                yield jd_block
                self.session_manager.update_session(session_id, {"jd_auto_suggested": True})

            # 分层引导问题（报告解读后的后续对话）
            if action != "解读报告":  # 解读报告本身已含 Layer 1 引导
                # 重新获取 session 拿到最新 guidance 状态
                session = self.session_manager.get_session(session_id)
                guidance = self._build_guidance_questions(session)
                if guidance:
                    full_reply += guidance
                    yield guidance
                    rounds = session.get("post_report_rounds", 0)
                    self.session_manager.update_session(session_id, {
                        "post_report_rounds": rounds + 1,
                    })

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
