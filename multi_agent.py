"""
===========================================
多 Agent 子模块 (Multi-Agent Sub-Agents)
===========================================
每个 Agent 专注一项任务，拥有独立且详尽的 Prompt。
所有 Agent 采用单轮或短轮对话，上下文永远干净，行为高度稳定。

架构：
┌──────────────────────────────┐
│  ChatAgent (Orchestrator)     │  路由 + 会话管理
└──────┬───────┬───────┬───────┘
       │       │       │
  诊断Agent  优化Agent  报告Agent
  (开场)    (优化)    (总结)
"""

import json
from typing import Optional, Generator, List, Dict
from openai import OpenAI


# ===========================================
# OptimizeAgent 工具定义（OpenAI Function Calling 格式）
# ===========================================

OPTIMIZE_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "搜索招聘网站上的岗位信息。当用户想了解市场上有哪些相关岗位、想看看目标岗位的招聘要求时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词，通常是岗位名称，如'产品经理'、'数据分析'"
                    },
                    "city": {
                        "type": "string",
                        "description": "目标城市，如'上海'、'北京'、'深圳'，默认'全国'"
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "re_evaluate_resume",
            "description": "用 HAY 评估体系重新评估优化后的简历，生成新的职级、薪酬、能力评分，并与原始评估做对比。当用户想看看优化后简历的评估变化时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "optimized_resume_text": {
                        "type": "string",
                        "description": "优化后的完整简历文本"
                    }
                },
                "required": ["optimized_resume_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_with_jd",
            "description": "将用户的简历与指定的 JD（岗位描述）进行匹配度分析，找出优势和差距。当用户提供了一个 JD 并想知道自己简历与之匹配程度时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "JD（岗位描述）的完整文本"
                    }
                },
                "required": ["jd_text"]
            }
        }
    }
]


# ===========================================
# 诊断 Agent（开场阶段专用）
# ===========================================

class DiagnosisAgent:
    """
    诊断 Agent —— 对照 HAY 8 因素逐项分析简历短板

    职责：
    - 分析评测结果，找出亮点和短板
    - 将 HAY 因素翻译为用户可理解的语言
    - 提出 2-3 个可优化方向供用户选择
    - 输出友好、专业的开场诊断

    设计原则：
    - 单轮对话，输入评测结果+简历，输出诊断开场白
    - Prompt 极其详尽（不用担心影响其他 Agent）
    - 温度偏低（0.3），确保诊断稳定一致
    """

    SYSTEM_PROMPT = """你是一位亲和力很强的简历优化顾问。

【你的任务】
生成一段友好的欢迎语，让用户感到轻松，并了解你能帮他做什么。

【欢迎语结构（严格按照以下结构输出）】
1. 一句亲切的欢迎（如"嗨，欢迎来到简历优化工坊～"）
2. 一句话说明你已经看过评测结果和简历了
3. 介绍你能帮用户做的 3 件事（用 emoji + 加粗，每个一句话简单说明）：
   - 简历优化：帮用户把简历写得更有竞争力
   - 岗位搜索：搜索市场上的岗位，了解招聘要求
   - JD 匹配分析：对比简历和目标岗位的匹配度
4. 一句话收尾，问用户想从哪里开始

【关键规则】
- 不要分析简历内容，不要提任何评测细节
- 不要提及"HAY"、"八因素"、"岗位评估"等方法论术语
- 总字数 80-120 字，不要太长
- 不要用 markdown 标题

【语气风格】
- 像一个热情的学长/学姐，亲切自然不做作
- 轻松活泼，可以用 emoji（但不要太多，2-3个即可）
- 口语化，有温度"""

    TEMPERATURE = 0.3

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def diagnose(self, assessment_context: dict, resume_text: str) -> str:
        """
        生成开场诊断（非流式）

        Args:
            assessment_context: 评测结果
            resume_text: 简历原文

        Returns:
            诊断开场白文本
        """
        user_prompt = self._build_input(assessment_context, resume_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.TEMPERATURE,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[DiagnosisAgent] 诊断生成失败: {e}")
            return "你好！我已经看过你的评测结果和简历了。准备好了就告诉我，我们可以开始优化简历。"

    def _build_input(self, ctx: dict, resume_text: str) -> str:
        """构建诊断 Agent 的输入"""
        factors = ctx.get("factors", {})
        abilities = ctx.get("abilities", {})
        grade = ctx.get("grade", "未知")
        salary = ctx.get("salaryRange", "未知")
        job_title = ctx.get("jobTitle", "未知")
        job_function = ctx.get("jobFunction", "未知")

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

        abilities_text = "\n".join(
            f"  - {name}: {info.get('score', '?')}分 ({info.get('level', '?')})"
            for name, info in abilities.items()
        ) if isinstance(abilities, dict) else "  暂无能力数据"

        resume_preview = resume_text[:3000] if len(resume_text) > 3000 else resume_text

        # 找出最弱的能力维度
        weak_abilities = []
        strong_abilities = []
        if isinstance(abilities, dict):
            sorted_abs = sorted(abilities.items(), key=lambda x: x[1].get("score", 50))
            weak_abilities = [name for name, _ in sorted_abs[:2]]
            strong_abilities = [name for name, _ in sorted_abs[-2:]]

        return f"""请对以下用户的评测结果和简历进行诊断分析。

【评测结果】
- 目标岗位: {job_title}
- 所属职能: {job_function}
- 评估职级: {grade}
- 薪酬区间: {salary}

HAY 8因素档位:
{factors_text}

5维能力得分:
{abilities_text}

能力分析提示:
- 较强维度: {', '.join(strong_abilities) if strong_abilities else '暂无'}
- 待提升维度: {', '.join(weak_abilities) if weak_abilities else '暂无'}

【简历内容】
{resume_preview}

请生成开场诊断。"""


# ===========================================
# 优化 Agent（优化阶段专用）
# ===========================================

class OptimizeAgent:
    """
    优化 Agent —— 专精简历段落改写

    职责：
    - 根据用户请求改写简历的具体段落
    - 给出修改前后对比
    - 说明每条修改对 HAY 哪个因素有提升
    - 主动追问缺失信息，绝不编造

    设计原则：
    - 每次调用接收完整上下文（简历+评测+对话摘要+用户请求）
    - Prompt 包含大量改写示例和技巧
    - 温度适中（0.5），平衡创造性和稳定性
    """

    SYSTEM_PROMPT = """你是一位经验丰富的简历优化顾问，擅长帮用户把简历写得更有竞争力，让简历更好地体现个人真实能力。

【你的核心能力】
你擅长将平淡的简历描述改写为有说服力的、能体现候选人真实价值的文字。

【改写方法（内部参考，不要向用户展示方法论名称）】

1. 背景-任务-行动-成果 结构强化：
   ❌ "负责用户运营工作"
   ✅ "在用户规模从10万增长到50万的阶段，独立负责用户分层运营体系搭建，通过设计3套差异化运营策略，推动核心用户月活提升23%"

2. 量化数据注入：
   - 能量化的一定要量化：用户数、营收、效率提升、团队规模
   - 没有精确数字用范围：「数十个」「百万级」「提升约20%」
   - 如果用户没提供数据，主动追问，绝不编造

3. 动词升级：
   - 执行层面 → 负责/主导/独立完成
   - 参与 → 核心参与/深度参与/作为关键成员
   - 帮助 → 推动/促进/赋能

4. 能力维度锚定（每条建议用通俗语言说明提升了什么能力）：
   - 加量化数据 → 让人看到你的 **业务影响力**
   - 写思考过程 → 体现你的 **分析和解决问题的能力**
   - 加团队协作 → 突出你的 **协作和沟通能力**
   - 突出专业深度 → 展示你的 **专业功底**
   - 写独立决策 → 展现你的 **独立负责能力**

5. 红线（绝对不可以做的事）：
   - ❌ 编造用户没有的经历
   - ❌ 编造具体数字（如果用户没提供）
   - ❌ 改变事实（把实习说成全职、把参与说成主导）
   - ✅ 如果需要数据但用户没提供，用「[待补充: 具体数字]」标记

【对话节奏——最高优先级规则，必须严格遵守】

判断用户意图，按类型回应：

A. 打招呼（"你好"、"hi"、"嗨"等）→ 简短回应 + 问用户想做什么。禁止：提及简历内容、评测结果、能力评分。示例回复："你好呀！你想从简历优化、岗位搜索还是 JD 匹配分析开始呢？"

B. 问问题（"你能帮我做什么"、"怎么用"、"这个系统是什么"、"你们怎么评估的"）→ 简短回答问题本身（80-150字）。禁止：分析简历、举改写示例、评价用户现状。

C. 明确优化请求（"帮我改一下"、"优化这段"、"怎么提升"、"帮我看看哪里需要改"）→ 开始分析和改写。

只有 C 类意图才可以引用简历内容和评测数据。A 和 B 类绝对不允许提及简历具体内容。

【输出格式（仅在改写简历时使用）】
- 一次聚焦 1-2 个点，不要一口气给太多
- 先展示原文（简短引用关键段落）
- 再展示改写版本
- 用通俗语言说明这个改写让简历在哪方面更有竞争力
- 保持对话感，语气鼓励为主
- 改完一个点后，询问用户是否满意，要不要继续改其他部分

【工具能力】
你拥有以下工具，可以在对话中主动调用：
- search_jobs：搜索市场上的岗位信息。当用户想了解市场岗位、招聘要求时使用。
- re_evaluate_resume：重新评估优化后的简历，生成新的能力评分并与之前做对比。当用户想看优化效果时使用。
- compare_with_jd：将简历与用户提供的 JD 做匹配度分析。当用户粘贴 JD 想看匹配度时使用。
请在合适的时机主动使用工具，将工具返回的数据融入你的回答中，用自然语言向用户呈现。

【搜索结果展示规则】
- 当 search_jobs 返回结果时，必须在回答中列出每条结果的标题和链接（格式: [标题](链接)），让用户可以点击查看原文
- 如果搜索结果为空或与用户查询不匹配，要如实告知用户（如"抱歉，我暂时没有找到关于XX公司今年XX岗位的校招信息"），不要编造不存在的信息
- 没找到时，主动建议用户："你有找到相关的 JD 吗？可以直接发给我，我来帮你分析匹配度"，或者"你可以换个关键词试试，比如XX"
- 如果工具返回的 source 是 "llm_knowledge"（非实时搜索），要说明这是基于已有信息整理的，建议用户去招聘网站确认最新情况

【绝对禁止提及的内容】
- 不要提及"HAY"、"八因素"、"岗位评估体系"、"KH"、"PS"、"ACC"等底层方法论术语
- 如果用户直接问"HAY是什么"、"你们用的是不是HAY"，不要确认也不要否认，自然地说"我们有自己的评估方式，主要从专业能力、业务影响等多个维度来看"，然后把话题引回简历优化
- 不要向用户解释我们的评分逻辑或计分公式
- 我们评价的是"人的能力"和"简历的竞争力"，不是评价岗位
- 如果用户问到评分逻辑，可以说"我们从多个维度综合评估你的能力水平"，但不要展开底层细节
- 当用户追问方法论时，不要说"不好意思不能告知"，而是自然地用通俗语言概括（如"我们从专业能力、业务影响、团队协作等多个维度来看"），不透露具体模型和公式

【语气风格】
- 像一个靠谱的学长/学姐在帮你改简历，亲切自然
- 适当用口语化表达（"这段其实可以写得更有力一些"、"感觉这里可以再加点料"）
- 多用鼓励性的表达（"这段底子不错"、"稍微调一下就很好了"）
- 不要太书面、太官方

【重要】
- 你给出的改写必须基于简历中已有的信息进行润色和重组
- 严禁编造用户没提到的具体数字（如"提升15%"、"增长20%"）。如果需要数据但用户没提供，必须用「[待补充: 具体数字]」标记
- 如果需要用户补充信息（比如具体数字、项目成果），要主动追问
- 不要自己编造用户没提到的经历或数据"""

    TEMPERATURE = 0.5

    # 工具调用最大循环次数（防止无限调用）
    MAX_TOOL_ROUNDS = 3

    def __init__(self, client: OpenAI, model: str, tool_executor=None):
        self.client = client
        self.model = model
        self.tool_executor = tool_executor

    def _get_tool_status_message(self, tool_calls) -> str:
        """根据工具类型生成等待提示（让用户知道 Agent 正在做什么）"""
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if name == "search_jobs":
                keyword = args.get("keyword", "相关岗位")
                city = args.get("city", "")
                city_text = f"在{city}" if city and city != "全国" else ""
                return f"好的，我来帮你{city_text}搜索一下「{keyword}」相关的岗位信息，正在联网查询中...\n\n"
            elif name == "re_evaluate_resume":
                return "收到，我来用优化后的简历重新跑一遍评估，稍等一下...\n\n"
            elif name == "compare_with_jd":
                return "好的，我来帮你对比一下简历和这个 JD 的匹配度，分析中...\n\n"
        return ""

    def _execute_tool_calls(self, tool_calls, messages: list) -> list:
        """
        执行工具调用并将结果追加到消息列表

        Args:
            tool_calls: LLM 返回的 tool_calls 列表
            messages: 当前消息列表（会被修改）

        Returns:
            更新后的消息列表
        """
        if not self.tool_executor:
            return messages

        # 先追加 assistant 的 tool_calls 消息
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in tool_calls
            ]
        }
        messages.append(assistant_msg)

        # 逐个执行工具并追加结果
        for tc in tool_calls:
            func_name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            print(f"[OptimizeAgent] 调用工具: {func_name}({arguments})")
            result = self.tool_executor.execute(func_name, arguments)
            print(f"[OptimizeAgent] 工具结果: {result[:200]}...")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        return messages

    def optimize(
        self,
        assessment_context: dict,
        resume_text: str,
        user_message: str,
        conversation_summary: str = "",
        recent_messages: List[dict] = None,
        memory_context: str = "",
    ) -> Generator[str, None, None]:
        """
        流式生成优化建议（支持 Function Call）

        策略：先非流式调用检测是否触发工具，
        如果触发则执行工具后再流式输出最终回答；
        如果未触发则直接流式输出。

        Yields:
            逐块文本
        """
        messages = self._build_messages(
            assessment_context, resume_text, user_message,
            conversation_summary, recent_messages or [], memory_context
        )

        try:
            has_tools = bool(self.tool_executor)
            tools_param = OPTIMIZE_AGENT_TOOLS if has_tools else None

            if has_tools:
                # 阶段 1：非流式调用，检测是否触发工具
                for round_idx in range(self.MAX_TOOL_ROUNDS):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.TEMPERATURE,
                        tools=tools_param,
                    )

                    choice = response.choices[0]

                    if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                        # 有工具调用 → 先输出状态提示，再执行
                        print(f"[OptimizeAgent] 第 {round_idx + 1} 轮工具调用")
                        status_msg = self._get_tool_status_message(choice.message.tool_calls)
                        if status_msg:
                            yield status_msg
                        messages = self._execute_tool_calls(choice.message.tool_calls, messages)
                        continue
                    else:
                        # 无工具调用 → 直接输出本次结果的文本内容
                        content = choice.message.content or ""
                        if content:
                            yield content
                        return

                # 超过最大轮次，做最后一次流式调用（不带 tools，强制文本输出）
                print(f"[OptimizeAgent] 达到最大工具调用轮次，强制文本输出")

            # 阶段 2（无工具 或 工具循环后）：流式输出
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.TEMPERATURE,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"[OptimizeAgent] 优化生成失败: {e}")
            yield "抱歉，生成优化建议时遇到了问题，请再试一次。"

    def optimize_sync(
        self,
        assessment_context: dict,
        resume_text: str,
        user_message: str,
        conversation_summary: str = "",
        recent_messages: List[dict] = None,
        memory_context: str = "",
    ) -> str:
        """非流式生成优化建议（支持 Function Call）"""
        messages = self._build_messages(
            assessment_context, resume_text, user_message,
            conversation_summary, recent_messages or [], memory_context
        )

        try:
            has_tools = bool(self.tool_executor)
            tools_param = OPTIMIZE_AGENT_TOOLS if has_tools else None

            for round_idx in range(self.MAX_TOOL_ROUNDS):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.TEMPERATURE,
                    tools=tools_param if has_tools else None,
                )

                choice = response.choices[0]

                if has_tools and choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                    print(f"[OptimizeAgent] 第 {round_idx + 1} 轮工具调用（sync）")
                    messages = self._execute_tool_calls(choice.message.tool_calls, messages)
                    continue
                else:
                    return (choice.message.content or "").strip()

            # 超过最大轮次，最后一次不带 tools
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.TEMPERATURE,
            )
            return (response.choices[0].message.content or "").strip()

        except Exception as e:
            print(f"[OptimizeAgent] 优化生成失败: {e}")
            return "抱歉，生成优化建议时遇到了问题，请再试一次。"

    def _build_messages(
        self,
        ctx: dict,
        resume_text: str,
        user_message: str,
        conversation_summary: str,
        recent_messages: List[dict],
        memory_context: str,
    ) -> list:
        """
        构建发送给 LLM 的消息列表

        结构：
        1. system prompt
        2. 上下文注入（作为第一条 user message）
        3. 历史对话（去掉当前这轮，因为当前消息单独传入）
        4. 当前用户消息

        注意：recent_messages 可能已包含当前用户消息（因为协调者先 add_message 再路由），
        所以需要排除末尾与 user_message 相同的消息，避免重复。
        """
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # 构建上下文
        context = self._build_context(ctx, resume_text, conversation_summary, memory_context)

        # 去掉 recent_messages 末尾的当前用户消息（避免重复）
        history = list(recent_messages) if recent_messages else []
        if history and history[-1].get("role") == "user" and history[-1].get("content") == user_message:
            history = history[:-1]

        # 注入上下文 + 历史对话
        if history:
            # 上下文作为独立的 user message 注入
            messages.append({"role": "user", "content": context})
            messages.append({"role": "assistant", "content": "好的，我已了解你的评测结果和简历内容。"})
            # 添加历史对话（保持 user/assistant 交替）
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            # 没有历史，上下文和当前消息合并
            messages.append({"role": "user", "content": context})
            messages.append({"role": "assistant", "content": "好的，我已了解你的评测结果和简历内容。"})

        # 当前用户消息
        messages.append({"role": "user", "content": user_message})

        return messages

    def _build_context(
        self, ctx: dict, resume_text: str,
        conversation_summary: str, memory_context: str
    ) -> str:
        """构建上下文信息"""
        factors = ctx.get("factors", {})
        abilities = ctx.get("abilities", {})

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

        abilities_text = "\n".join(
            f"  - {name}: {info.get('score', '?')}分 ({info.get('level', '?')})"
            for name, info in abilities.items()
        ) if isinstance(abilities, dict) else "  暂无"

        resume_preview = resume_text[:3000] if len(resume_text) > 3000 else resume_text

        parts = []

        if conversation_summary:
            parts.append(f"<conversation_summary>\n{conversation_summary}\n</conversation_summary>")

        parts.append(f"""<assessment>
目标岗位: {ctx.get('jobTitle', '未知')} | 职能: {ctx.get('jobFunction', '未知')}
职级: {ctx.get('grade', '未知')} | 薪酬: {ctx.get('salaryRange', '未知')}

HAY 8因素:
{factors_text}

5维能力:
{abilities_text}
</assessment>""")

        parts.append(f"<resume>\n{resume_preview}\n</resume>")

        if memory_context:
            parts.append(f"<memory>\n{memory_context}\n</memory>")

        return "\n\n".join(parts)


# ===========================================
# 报告 Agent（总结阶段专用）
# ===========================================

class ReportAgent:
    """
    报告 Agent —— 生成优化总结报告

    职责：
    - 汇总本次对话中的所有优化内容
    - 评估优化后的能力维度提升预期
    - 给出后续建议
    - 鼓励用户

    设计原则：
    - 单轮对话，输入全部会话上下文，输出完整总结
    - 温度低（0.3），确保总结准确稳定
    """

    SYSTEM_PROMPT = """你是一位亲切的简历优化顾问，负责在对话结束时帮用户做一个轻松的小总结。

【总结要求】

1. 聊聊这次改了啥：
   - 简单列出本次对话中改了哪几个地方
   - 用通俗语言说说每个改动让简历在哪方面更有竞争力

2. 效果预期：
   - 用通俗语言说说优化后简历整体会有什么提升
   - 不要编造分数，说感受就好（如"专业度会明显提升"）

3. 后续建议：
   - 给 1-2 个下次还可以继续打磨的方向

4. 收尾鼓励：
   - 轻松正能量收尾

【禁止事项】
- 不要提及"HAY"、"八因素"、"岗位评估"等底层方法论术语
- 我们评价的是人的能力和简历竞争力，不是评价岗位

【格式和语气】
- 总字数 150-250 字
- 像朋友聊天一样自然轻松
- 用 **加粗** 高亮关键词
- 不使用 markdown 标题，用自然语言组织
- 不要像写报告，就是聊天收尾的感觉"""

    TEMPERATURE = 0.3

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def generate_report(
        self,
        assessment_context: dict,
        resume_text: str,
        conversation_summary: str = "",
        memory_context: str = "",
        recent_messages: List[dict] = None,
    ) -> Generator[str, None, None]:
        """
        流式生成总结报告

        Yields:
            逐块文本
        """
        user_prompt = self._build_input(
            assessment_context, resume_text,
            conversation_summary, memory_context, recent_messages or []
        )

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.TEMPERATURE,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"[ReportAgent] 报告生成失败: {e}")
            yield "感谢你的耐心配合！本次优化已完成，建议你根据我们讨论的方向继续完善简历。加油！"

    def generate_report_sync(
        self,
        assessment_context: dict,
        resume_text: str,
        conversation_summary: str = "",
        memory_context: str = "",
        recent_messages: List[dict] = None,
    ) -> str:
        """非流式生成总结报告"""
        user_prompt = self._build_input(
            assessment_context, resume_text,
            conversation_summary, memory_context, recent_messages or []
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.TEMPERATURE,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ReportAgent] 报告生成失败: {e}")
            return "感谢你的耐心配合！本次优化已完成，建议你根据我们讨论的方向继续完善简历。加油！"

    def _build_input(
        self, ctx: dict, resume_text: str,
        conversation_summary: str, memory_context: str,
        recent_messages: List[dict]
    ) -> str:
        """构建报告 Agent 的输入"""
        parts = []

        parts.append(f"""【评测概况】
目标岗位: {ctx.get('jobTitle', '未知')} | 职能: {ctx.get('jobFunction', '未知')}
评估职级: {ctx.get('grade', '未知')} | 薪酬区间: {ctx.get('salaryRange', '未知')}""")

        if conversation_summary:
            parts.append(f"【对话过程摘要】\n{conversation_summary}")

        if memory_context:
            parts.append(f"【结构化记忆】\n{memory_context}")

        # 最近几轮对话（提供即时上下文）
        if recent_messages:
            recent_text = "\n".join(
                f"{'用户' if m['role'] == 'user' else '顾问'}: {m['content'][:200]}"
                for m in recent_messages[-6:]
            )
            parts.append(f"【最近对话】\n{recent_text}")

        parts.append("请生成本次优化的总结报告。")

        return "\n\n".join(parts)
