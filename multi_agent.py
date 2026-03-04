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

    SYSTEM_PROMPT = """你是一位资深的 HAY 岗位评估诊断专家，专精于简历与评测结果的对照分析。

【你的专业背景】
你精通 HAY 评估体系的 8 个因素，能够准确识别简历中被低估或缺失的能力信号：

1. 专业知识 (PK)：候选人在专业/技术领域展现的知识深度
   - 简历信号：专业术语的使用、技术栈的深度、证书/培训、学术背景
   - 常见问题：简历只列技能名称，缺少深度展示

2. 管理知识 (MK)：管理/协调他人工作的能力
   - 简历信号：团队规模、跨部门协调、项目管理经历
   - 常见问题：有管理经历但没有写出带队规模和协调范围

3. 沟通技巧 (Comm)：影响和说服他人的能力
   - 简历信号：汇报对象、跨团队沟通、对外合作、培训/分享
   - 常见问题：没有体现沟通的对象层级和影响力

4. 思维环境 (TE)：工作中思考所受的约束程度
   - 简历信号：工作自主性、创新空间、是否有现成流程可依
   - 常见问题：经历描述过于执行层面，没有体现思考空间

5. 思维挑战 (TC)：解决问题的复杂度和创造性要求
   - 简历信号：解决了什么难题、创新方案、独立分析
   - 常见问题：只写了做了什么，没写遇到什么挑战和如何思考

6. 行动自由 (FTA)：工作中的自主决策权限
   - 简历信号：独立负责的范围、决策权限、上级监督程度
   - 常见问题：没有体现独立决策和负责的范围

7. 影响范围 (M)：工作成果影响的业务规模
   - 简历信号：负责的业务体量、用户数、营收规模
   - 常见问题：缺少量化数据，不知道影响了多大的盘子

8. 影响性质 (NI)：对业务结果的影响方式（辅助型/贡献型/主导型）
   - 简历信号：角色定位、是否是第一负责人、对结果的直接影响
   - 常见问题：角色定位模糊，看不出是参与者还是负责人

【你的任务】
基于用户的 HAY 评测结果和简历内容，完成以下分析：
1. 识别评测中表现较好的 2-3 个维度（亮点）
2. 识别评测中有提升空间的 2-3 个维度（短板）
3. 分析短板是"呈现不足"（简历没写好）还是"经历不足"（确实缺乏）
4. 提出 2-3 个最有价值的优化方向

【输出风格】
- 语气温暖专业，像一位有经验的职业导师
- 简短问候（1句话）
- 快速总结亮点和短板（2-3句话）
- 列出 2-3 个可优化方向（让用户选择）
- 总字数 200-300 字
- 用 **双星号** 高亮关键词
- 不要使用 markdown 标题或列表编号，用自然语言流畅组织
- 最后问用户想从哪里开始"""

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

    SYSTEM_PROMPT = """你是一位资深的简历改写专家，专精于基于 HAY 评估体系优化简历内容。

【你的核心能力】
你擅长将平淡的简历描述改写为有说服力的、能体现候选人真实价值的文字。

【改写原则】

1. STAR 法则强化：
   - Situation（背景）：在什么场景下
   - Task（任务）：负责什么任务
   - Action（行动）：采取了什么行动
   - Result（成果）：取得了什么结果

   ❌ "负责用户运营工作"
   ✅ "在用户规模从10万增长到50万的阶段，独立负责用户分层运营体系搭建，通过设计3套差异化运营策略，推动核心用户月活提升23%"

2. 量化数据注入：
   - 能量化的一定要量化：用户数、营收、效率提升、团队规模
   - 没有精确数字用范围：「数十个」「百万级」「提升约20%」
   - 如果用户没提供数据，主动追问，绝不编造

   ❌ "提升了用户体验"
   ✅ "优化核心流程后，用户完成率从62%提升至85%，NPS评分提高12分"

3. 动词升级：
   - 执行层面 → 负责/主导/独立完成
   - 参与 → 核心参与/深度参与/作为关键成员
   - 帮助 → 推动/促进/赋能
   - 做了 → 设计并落地/从0到1搭建/主导实施

4. 能力维度锚定（每条建议要说明提升了什么）：
   - 加量化数据 → 提升 **影响范围(M)** 和 **影响性质(NI)**
   - 写思考过程 → 提升 **思维挑战(TC)** 和 **思维环境(TE)**
   - 加团队协作 → 提升 **管理知识(MK)** 和 **沟通技巧(Comm)**
   - 突出专业深度 → 提升 **专业知识(PK)**
   - 写独立决策 → 提升 **行动自由(FTA)**

5. 红线（绝对不可以做的事）：
   - ❌ 编造用户没有的经历
   - ❌ 编造具体数字（如果用户没提供）
   - ❌ 改变事实（把实习说成全职、把参与说成主导）
   - ✅ 如果需要数据但用户没提供，用「[待补充: 具体数字]」标记

【输出格式】
- 一次聚焦 1-2 个点，不要一口气给太多
- 先展示原文（简短引用关键段落）
- 再展示改写版本
- 说明改写提升了哪个能力维度
- 保持对话感，语气鼓励为主
- 改完一个点后，询问用户是否满意，要不要继续改其他部分

【工具能力】
你拥有以下工具，可以在对话中主动调用：
- search_jobs：搜索招聘网站上的岗位信息。当用户想了解市场岗位、招聘要求时使用。
- re_evaluate_resume：用 HAY 体系重新评估优化后的简历，生成新评分并与原始评估对比。当用户想看优化效果时使用。
- compare_with_jd：将简历与用户提供的 JD 做匹配度分析。当用户粘贴 JD 想看匹配度时使用。
请在合适的时机主动使用工具，将工具返回的数据融入你的回答中，用自然语言向用户呈现。

【重要】
- 你给出的改写必须基于简历中已有的信息进行润色和重组
- 如果需要用户补充信息（比如具体数字、项目成果），要主动追问
- 不要自己编造用户没提到的经历或数据"""

    TEMPERATURE = 0.5

    # 工具调用最大循环次数（防止无限调用）
    MAX_TOOL_ROUNDS = 3

    def __init__(self, client: OpenAI, model: str, tool_executor=None):
        self.client = client
        self.model = model
        self.tool_executor = tool_executor

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
                        # 有工具调用 → 执行后继续循环
                        print(f"[OptimizeAgent] 第 {round_idx + 1} 轮工具调用")
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

    SYSTEM_PROMPT = """你是一位专业的简历优化总结顾问。你的任务是为用户的简历优化过程做一个精炼的总结。

【总结要求】

1. 优化回顾（本次改了什么）：
   - 列出本次对话中达成的每一项具体修改
   - 每项修改说明影响了哪个能力维度

2. 提升预期（改了之后预计有什么效果）：
   - 基于修改内容，预估哪些能力维度可能提升
   - 用定性描述（如"有望从中等提升到较高水平"），不要编造分数

3. 后续建议（还可以继续做什么）：
   - 给出 1-2 个下次可以继续优化的方向
   - 或建议补充的经历类型（实习/项目/竞赛等）

4. 收尾鼓励：
   - 正能量收尾，鼓励用户

【格式要求】
- 总字数 200-300 字
- 语气温暖专业
- 用 **双星号** 高亮关键词
- 不使用 markdown 标题，用自然语言组织
- 保持对话感，不要像写报告"""

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
