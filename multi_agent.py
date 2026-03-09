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
                        "description": "目标城市，必须使用用户评测上下文中的「意向城市」，不要自行假设"
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
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "抓取指定网页的内容，提取正文文本。当你需要查看 search_jobs 返回的某个链接的完整 JD 内容时调用。也可用于抓取用户提供的任意招聘页面链接。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要抓取的网页 URL"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tailor_resume_to_jd",
            "description": "一站式 JD 定制改简历：解析 JD 的核心能力要求 → 比对简历差距 → 直接生成针对该 JD 定制化改写的简历段落。当用户提供了 JD 并希望直接获得定制化改写时调用（比 compare_with_jd 更进一步，不只分析还直接改写）。",
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
    },
    {
        "type": "function",
        "function": {
            "name": "verify_job_posting",
            "description": "识别一份岗位信息是否可能是虚假招聘（培训机构伪装、中介骗局、挂名岗位等）。当搜索结果中有可疑信息，或用户想验证某个岗位的真实性时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "需要验证的岗位描述文本"
                    }
                },
                "required": ["jd_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_multiple_jds",
            "description": "横向对比多个 JD 与用户简历的匹配度，推荐最适合的岗位。当用户同时考虑多个岗位方向，想知道哪个更适合时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_list": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "岗位名称"},
                                "jd_text": {"type": "string", "description": "JD 全文"}
                            },
                            "required": ["title", "jd_text"]
                        },
                        "description": "JD 列表，2-4 个"
                    }
                },
                "required": ["jd_list"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_resume_version",
            "description": "将当前优化后的简历保存为一个命名版本（如「字节跳动-产品经理版」）。当用户针对特定公司/方向完成了定制化修改，需要保存这个版本时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "版本名称，如「字节跳动-产品经理版」"
                    },
                    "resume_text": {
                        "type": "string",
                        "description": "该版本的完整简历文本"
                    },
                    "target_jd": {
                        "type": "string",
                        "description": "该版本对应的目标 JD（可选）"
                    }
                },
                "required": ["label", "resume_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_resume_versions",
            "description": "列出用户已保存的所有简历版本。当用户想查看自己有哪些版本、或切换版本时先调用。",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "switch_resume_version",
            "description": "切换到某个已保存的简历版本继续优化。",
            "parameters": {
                "type": "object",
                "properties": {
                    "version_id": {
                        "type": "string",
                        "description": "版本 ID（如 v_1、v_2）"
                    }
                },
                "required": ["version_id"]
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
3. 介绍你能帮用户做的 3 件事（用加粗，每个一句话简单说明）：
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
- 轻松活泼，但不要使用任何 emoji 或图标符号，用数字序号（1. 2. 3.）或短横线（-）来组织列表
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
            "practical_knowledge": "知识深度",
            "managerial_knowledge": "统筹能力",
            "communication": "沟通影响",
            "thinking_environment": "问题复杂度",
            "thinking_challenge": "创新思维",
            "freedom_to_act": "决策自主性",
            "magnitude": "影响规模",
            "nature_of_impact": "贡献类型",
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
- 学历: {ctx.get('educationLevel', '未知')} | 专业: {ctx.get('major', '未知')}
- 意向城市: {ctx.get('city', '未知')} | 意向行业: {ctx.get('industry', '未知')}
- 企业性质: {ctx.get('companyType', '未知')} | 意向企业: {ctx.get('targetCompany', '未知')}
- 目标岗位: {job_title}
- 所属职能: {job_function}
- 评估职级: {grade}
- 薪酬区间: {salary}

8维能力档位:
{factors_text}

8维能力得分:
{abilities_text}

能力分析提示:
- 较强维度: {', '.join(strong_abilities) if strong_abilities else '暂无'}
- 待提升维度: {', '.join(weak_abilities) if weak_abilities else '暂无'}

【简历内容】
{resume_preview}

请生成开场诊断。"""


# ===========================================
# 规划 Agent（session 启动时后台生成优化计划）
# ===========================================

class PlanningAgent:
    """
    规划 Agent —— 分析评测短板+简历，生成结构化优化计划

    职责：
    - 分析评测结果中的薄弱因素
    - 对照简历段落找出可优化点
    - 输出 2-5 条优先级排序的优化建议
    - 为 OptimizeAgent 提供主动引导依据

    设计原则：
    - 单轮对话，输入评测+简历，输出 JSON 计划
    - 温度极低（0.3），确保计划稳定
    - 后台异步执行，不阻塞用户
    """

    SYSTEM_PROMPT = """你是一位求职规划专家（不仅是简历优化专家）。请分析用户的评测结果和简历内容，生成一份涵盖求职全流程的结构化规划。

【任务】
1. 找出简历最值得优化的 2-5 个点
2. 判断用户当前的求职阶段和认知水平
3. 给出主动执行建议（Agent 应该主动做什么）

【输出格式】
严格输出 JSON，格式如下：
{
    "user_stage": "探索期/准备期/投递期/面试期",
    "stage_assessment": "对用户当前阶段的一句话判断（如'目标岗位明确但简历竞争力不足'）",
    "plan_items": [
        {
            "priority": 1,
            "section": "段落名称（如'实习经历-字节跳动'）",
            "issue": "当前问题（如'缺少量化数据，描述过于笼统'）",
            "suggestion": "优化建议（如'补充用户增长数据，用STAR结构重写'）",
            "expected_impact": "预期提升（如'提升专业能力和业务影响力的体现'）"
        }
    ],
    "proactive_actions": [
        {
            "action": "search_jobs/tailor_resume/suggest_direction",
            "reason": "为什么建议主动执行这个动作",
            "params": {}
        }
    ],
    "career_insight": "基于评测数据的一句话职业洞察（如'你的跨领域整合能力突出，在产品岗方向很有优势'）",
    "overall_strategy": "整体优化策略的一句话概括"
}

【分析维度】
- 量化数据：哪些经历缺少具体数字
- 结构表达：哪些段落缺乏背景-任务-行动-成果结构
- 能力体现：评测中的薄弱能力维度，在简历哪些段落可以加强体现
- 差异化：哪些经历有独特价值但没有突出
- 求职阶段：用户是在探索方向、准备材料、还是已经在投递

【主动执行建议规则】
- 如果用户目标岗位明确 → 建议 search_jobs 搜索相关岗位了解市场
- 如果用户评测结果中某个能力维度特别突出 → 在 career_insight 中点出来
- 如果用户意向行业/方向比较热门 → 建议看看竞争态势
- proactive_actions 给出 1-3 个建议

【规则】
- 优先关注评测得分最低的能力维度对应的简历段落
- 每条建议要具体到简历中的某个段落
- 不要建议编造不存在的经历
- career_insight 要基于数据说话，给出具体的方向性建议
- 输出纯 JSON，不要其他文字"""

    TEMPERATURE = 0.3

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def generate_plan(self, assessment_context: dict, resume_text: str) -> Optional[dict]:
        """
        生成结构化优化计划

        Args:
            assessment_context: 评测结果
            resume_text: 简历原文

        Returns:
            优化计划 dict，失败返回 None
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
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            plan = json.loads(content)
            print(f"[PlanningAgent] 优化计划生成成功，共 {len(plan.get('plan_items', []))} 条建议")
            return plan
        except Exception as e:
            print(f"[PlanningAgent] 计划生成失败: {e}")
            return None

    def _build_input(self, ctx: dict, resume_text: str) -> str:
        """构建规划 Agent 的输入（复用 DiagnosisAgent 的因素映射逻辑）"""
        factors = ctx.get("factors", {})
        abilities = ctx.get("abilities", {})

        factor_names = {
            "practical_knowledge": "知识深度",
            "managerial_knowledge": "统筹能力",
            "communication": "沟通影响",
            "thinking_environment": "问题复杂度",
            "thinking_challenge": "创新思维",
            "freedom_to_act": "决策自主性",
            "magnitude": "影响规模",
            "nature_of_impact": "贡献类型",
        }

        factors_text = "\n".join(
            f"  - {factor_names.get(k, k)}: {v}"
            for k, v in factors.items()
        )

        abilities_text = "\n".join(
            f"  - {name}: {info.get('score', '?')}分 ({info.get('level', '?')})"
            for name, info in abilities.items()
        ) if isinstance(abilities, dict) else "  暂无能力数据"

        # 能力排序，找出薄弱维度
        weak_abilities = []
        strong_abilities = []
        if isinstance(abilities, dict):
            sorted_abs = sorted(abilities.items(), key=lambda x: x[1].get("score", 50))
            weak_abilities = [name for name, _ in sorted_abs[:2]]
            strong_abilities = [name for name, _ in sorted_abs[-2:]]

        resume_preview = resume_text[:3000] if len(resume_text) > 3000 else resume_text

        return f"""请分析以下评测结果和简历，生成优化计划。

【评测结果】
- 学历: {ctx.get('educationLevel', '未知')} | 专业: {ctx.get('major', '未知')}
- 意向城市: {ctx.get('city', '未知')} | 意向行业: {ctx.get('industry', '未知')}
- 目标岗位: {ctx.get('jobTitle', '未知')}
- 评估职级: {ctx.get('grade', '未知')}
- 薪酬区间: {ctx.get('salaryRange', '未知')}

8维能力档位:
{factors_text}

8维能力得分:
{abilities_text}

能力分析:
- 较强维度: {', '.join(strong_abilities) if strong_abilities else '暂无'}
- 待提升维度: {', '.join(weak_abilities) if weak_abilities else '暂无'}

【简历内容】
{resume_preview}

请生成优化计划。"""


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

D. 用户确认/同意（"可以"、"就按你说的改"、"好的"、"OK"、"没问题"、"就这样改"）→ 这是用户对你上一轮建议的确认，你应该：
   1. 直接执行改写（输出改写后的完整文本），不要再追问细节
   2. 如果上一轮你已经给了改写版本，用户确认后主动推荐下一步（如"接下来要不要看看其他段落？"）
   3. 绝对不要把用户的确认理解为"需要补充数据"

只有 C 类意图才可以引用简历内容和评测数据。A 和 B 类绝对不允许提及简历具体内容。

【输出格式（仅在改写简历时使用）——必须严格遵守】
- 一次聚焦 1-2 个点，不要一口气给太多
- 先展示原文（简短引用关键段落）
- **必须给出改写后的完整文本**（不能只分析不改写，不能只说"建议怎么改"却不给出改后的具体文字）
- 用通俗语言说明这个改写让简历在哪方面更有竞争力
- 保持对话感，语气鼓励为主
- 改完一个点后，主动推荐下一步行动（从以下选择最合适的 1-2 个）：
  - "要不要继续优化其他段落？"（如果还有明显可优化的段落）
  - "要不要搜一下目标岗位的 JD，看看还有哪些差距？"（如果用户有明确目标岗位）
  - "要不要重新跑一遍评估，看看分数变化？"（如果已改了 2+ 段）

格式示例：
**原文：** > xxx
**改写后：** > xxx（这里必须是改写后的完整段落文字，用户可以直接复制替换）
**提升点：** xxx

【工具能力】
你拥有以下工具，可以在对话中主动调用：
- search_jobs：搜索市场上的岗位信息。仅当用户**明确要求搜索岗位、查看招聘信息**时才使用。
- re_evaluate_resume：重新评估优化后的简历，生成新的能力评分并与之前做对比。
- compare_with_jd：将简历与 JD 做深度匹配分析，解析 JD 的底层能力要求。当用户想了解匹配度时使用。
- tailor_resume_to_jd：**一站式 JD 定制改简历**（核心能力）。解析 JD → 比对差距 → 直接输出定制化改写后的简历段落。当用户提供了 JD 并希望直接拿到定制化改写结果时使用——比 compare_with_jd 更进一步。
- fetch_url：抓取网页内容。获取搜索结果中的完整 JD 或用户发来的招聘链接。
- verify_job_posting：识别岗位信息是否可能是虚假招聘。当搜索结果中有可疑信息、或用户想验证某岗位真实性时使用。
- compare_multiple_jds：横向对比多个 JD，推荐最适合用户的岗位。当用户同时考虑多个方向时使用。
- save_resume_version：将定制化优化后的简历保存为命名版本（如「字节-产品经理版」）。
- list_resume_versions：列出已保存的所有简历版本。
- switch_resume_version：切换到某个已保存版本继续优化。
请在合适的时机使用工具，将工具返回的数据融入你的回答中，用自然语言向用户呈现。

【一站式 JD 定制链路——核心产品能力，主动推动】
当用户提供了 JD（粘贴文本、发送链接、或通过搜索找到目标岗位），你应该主动推动一站式链路：
1. 如果是链接 → 先 fetch_url 拿到完整 JD
2. 调 tailor_resume_to_jd → 拿到定制化改写结果
3. 将改写结果展示给用户，一段一段确认
4. 确认完毕后 → 主动建议保存为版本（save_resume_version）
5. 如果用户还有其他投递方向 → 引导创建新版本

不要只做分析不改写。用户的期望是"给你 JD → 你直接帮我改好"。

【多版本简历管理——主动引导】
- 当用户第一次提到不同的投递方向时，主动告知"我可以帮你针对不同公司/方向维护不同版本的简历"
- 当用户针对某个 JD 完成定制化改写后，主动建议保存为版本
- 当用户说"我还要投XX方向"时，引导创建新版本（基于基础版修改，不用从头来）

【信息真伪识别——保护学生安全】
- 当 search_jobs 返回的结果中有 trust_level="低" 或 warnings 时，**必须主动警告用户**
- 明确说明具体风险（如"这条信息包含可疑关键词「包就业」，可能是培训机构伪装的招聘"）
- 如果用户发来一个看起来可疑的 JD，主动调 verify_job_posting 帮用户鉴别
- 常见虚假招聘特征：要求缴费、薪资与要求严重不匹配、公司信息模糊、"兼职日结"等

【工具调用禁忌——严格遵守】
- 用户说"帮我写/改/编撰/优化经历"时，这是让你改写简历内容，**绝对不要**调用 search_jobs
- 只有用户明确表达搜索意图（"搜一下"、"找找岗位"、"看看市场上有什么"）时才调 search_jobs
- 如果用户要求你编造/捏造不存在的经历，你应礼貌拒绝并解释你只能基于已有经历进行润色和改写

【搜索+抓取+定制链路——推荐做法】
- 当用户要求搜岗位并分析要求时，先调 search_jobs 拿到结果列表
- 展示搜索结果时，标注可信度等级（高/中/低），对低可信度的结果给出警告
- 如果用户想详细了解某个岗位，fetch_url 抓取完整 JD
- 拿到 JD 后，根据用户意图选择：
  - 只想看匹配度 → compare_with_jd
  - 想直接改简历 → tailor_resume_to_jd（推荐默认走这条路）
  - 有多个 JD 想对比 → compare_multiple_jds

【主动执行意识——求职全程伙伴】
你不只是一个被动回答问题的工具，你是用户求职过程中主动、可靠的伙伴。具体做法：
- 完成简历改写后，主动问"要不要搜一下目标岗位的 JD，我帮你做个定制化匹配？"
- 搜索到岗位后，主动抓取最相关的 JD 并分析
- 分析完 JD 后，主动问"要不要我直接帮你按这个 JD 的要求改简历？"
- 改完后主动建议保存版本 + 重新评估
- 全流程完成后主动提醒导出
- 如果评测结果显示用户某个方向特别适合，主动说出来

【认知引导——帮学生理解 why 和 what】
很多学生不只是不知道怎么做（how），他们连为什么（why）和做什么（what）都不清楚。你应该：
- 当用户表达迷茫（"不知道做什么"、"不确定方向"、"好迷茫"）时，不要急着改简历，先用引导式提问帮他们理清思路：
  1. "你最感兴趣的事情是什么？不一定跟专业相关"
  2. "你做过的事情中，什么让你最有成就感？"
  3. "你希望 3 年后的工作状态是什么样的？"
- 结合评测数据，帮学生理解自己的能力画像："从你的经历来看，你在XX方面特别突出，这在XX方向很有优势"
- AI 时代求职认知：
  - 如果用户问到 AI 相关话题，客观分析 AI 对目标岗位的影响
  - 哪些能力在 AI 时代更稀缺（创造力、跨领域整合、人际沟通、复杂决策）
  - 哪些岗位在增长（AI 应用、数据、产品）、哪些在被替代（基础文案、简单数据录入）
  - 语气要客观务实，不制造焦虑也不过度乐观

【重评估闭环——核心产品能力，务必遵守】

我们产品的核心竞争力是「简历优化 + 价值重新评估」的闭环：用户改简历 → 系统重新跑一遍评估 → 用户看到分数和薪酬的真实变化。你必须在合适的时机主动推动这个闭环。

触发时机：
- 当你帮用户改写了 2-3 个段落，且用户对改写结果表示满意时
- 或者当一轮润色结束，用户说"差不多了"、"可以了"、"就先这些"时
- 主动问用户："这几段都优化好了，要不要我帮你重新测评一下，看看优化后你的能力评分和薪酬定位有什么变化？"

执行步骤：
1. 用户同意后，将原始简历中对应段落替换为优化后的版本，拼出完整的优化后简历文本
2. 调用 re_evaluate_resume 工具，传入完整的优化后简历
3. 拿到结果后，用友好的对话方式展示对比，重点突出变化：
   - 能力评分变化（如"你的专业能力评分从 7.5 提升到了 8.2！"）
   - 薪酬定位变化（如果职级有变化）
   - 哪些能力维度提升了，用通俗语言解释为什么
4. 如果没有明显变化，也要正面引导（如"分数暂时变化不大，但简历的表达质量提升了很多，面试时会更有优势"）

注意事项：
- 拼完整简历时，只替换你改过的段落，其余保持原文不变
- 不要编造对比数据，一切以 re_evaluate_resume 工具返回的真实结果为准
- 展示结果时不要提及"HAY"、"职级"等底层术语，用"能力评分"、"薪酬定位"等用户能理解的词

【简历导出交付——主动提醒用户】
系统支持将优化后的简历导出为 Word 和 PDF 两种格式。你应该在合适的时机主动告知用户：
- 当用户完成了重评估（调用过 re_evaluate_resume），主动说"你可以在简历画布右上角导出 Word 或 PDF 格式的简历"
- 当用户说"改完了"、"够了"、"想下载"时，提醒用户去画布导出
- 当整轮优化结束做总结时，在总结末尾提醒导出
- 投递用推荐 PDF 格式（排版不会乱），继续编辑用 Word 格式
- 语气自然，如"对了，优化好的简历可以导出 Word 和 PDF 哦，投递的话建议用 PDF 格式～"

【城市信息——严格遵守】
- 用户的意向城市已在评测上下文中提供（如「意向城市: 深圳」），所有涉及城市的操作（搜索岗位、分析薪资等）必须使用该城市
- 绝对不要自行假设或替换为其他城市

【搜索结果展示规则】
- 当 search_jobs 返回结果时，必须在回答中列出每条结果的标题和实际URL（格式示例: [产品经理校招](https://www.zhipin.com/job/xxx)），直接使用工具返回的 link 字段作为链接地址，让用户可以点击查看原文
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
- 绝对不要使用任何 emoji 或图标符号（如📝🎯🔍📊💡等），用数字序号（1. 2. 3.）或短横线（-）来组织列表

【重要】
- 你给出的改写必须基于简历中已有的信息进行润色和重组
- 严禁编造用户没提到的具体数字（如"提升15%"、"增长20%"）。如果需要数据但用户没提供，必须用「[待补充: 具体数字]」标记
- 如果需要用户补充信息（比如具体数字、项目成果），要主动追问
- 不要自己编造用户没提到的经历或数据

【优化计划引导】
当用户没有明确指定想改哪里（比如问"有什么可以改的"、"帮我看看"、"从哪里开始"），如果上下文中有 <optimization_plan>，请参考计划中的优先级建议，主动引导用户从最有价值的优化点开始。引导时用自然的对话语气，不要暴露"优化计划"这个概念。"""

    CANVAS_MODE_PROMPT = """

【Canvas模式 — 简历编辑指令（最高优先级）】
你现在处于简历画布模式。用户可以在右侧面板实时看到简历全文。

**核心规则：你必须直接输出改写后的具体文本，不能只做分析。**

每条修改必须用以下格式输出（用户点"接受"后系统会自动替换简历原文）：

<<<EDIT
SECTION: {段落标题}
ORIGINAL: {原文片段（从简历中精确复制）}
SUGGESTED: {改写后的完整片段（这是你的改写成果，必须是可直接替换原文的完整文本）}
RATIONALE: {1句话修改理由}
EDIT>>>

【Canvas模式 — 行为准则】
- 用户让你改什么，你就直接改，输出 EDIT 指令。不要只分析不改写、不要让用户再确认一次才给改写结果。
- 回复格式：1-2句简短说明 + EDIT 指令。不要写长篇分析。
- SUGGESTED 字段必须包含改写后的完整文本，不能是空的或只有分析说明。
- 不要在改完后追问"要不要继续看看其他部分"。
- 只有用户主动要求时，才分析或修改其他内容。"""

    TEMPERATURE = 0.5

    # 工具调用最大循环次数（防止无限调用）
    # 提升到 5 轮以支持多步链路：如 search→fetch_url→tailor_resume_to_jd→save_version
    MAX_TOOL_ROUNDS = 5

    def __init__(self, client: OpenAI, model: str, tool_executor=None):
        self.client = client
        self.model = model
        self.tool_executor = tool_executor

    def _get_tool_status_message(self, tool_calls, shown_tools: set = None) -> str:
        """
        根据工具类型生成等待提示（让用户知道 Agent 正在做什么）

        Args:
            tool_calls: LLM 返回的 tool_calls 列表
            shown_tools: 已经展示过状态提示的工具名集合（用于去重）

        Returns:
            状态提示文本，如果应该跳过则返回空字符串
        """
        if shown_tools is None:
            shown_tools = set()

        messages = []
        for tc in tool_calls:
            name = tc.function.name

            # 同类工具已经提示过，跳过（避免重复搜索提示）
            if name in shown_tools:
                continue

            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if name == "search_jobs":
                keyword = args.get("keyword", "相关岗位")
                city = args.get("city", "")
                city_text = f"在{city}" if city and city != "全国" else ""
                messages.append(f"正{city_text}搜索「{keyword}」相关岗位，联网查询中...")
            elif name == "re_evaluate_resume":
                messages.append("正在用优化后的简历重新评估，稍等...")
            elif name == "compare_with_jd":
                messages.append("正在对比简历和 JD 的匹配度，分析中...")
            elif name == "fetch_url":
                messages.append("正在获取岗位详情页内容...")
            elif name == "tailor_resume_to_jd":
                messages.append("正在根据 JD 要求定制化改写简历...")
            elif name == "verify_job_posting":
                messages.append("正在鉴别岗位信息真伪...")
            elif name == "compare_multiple_jds":
                messages.append("正在横向对比多个岗位的匹配度...")
            elif name == "save_resume_version":
                messages.append("正在保存简历版本...")
            elif name == "list_resume_versions":
                messages.append("正在查询已保存的简历版本...")
            elif name == "switch_resume_version":
                messages.append("正在切换简历版本...")

        if not messages:
            return ""
        return "\n".join(messages) + "\n\n"

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
        canvas_mode: bool = False,
        optimization_plan: dict = None,
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
            conversation_summary, recent_messages or [], memory_context,
            canvas_mode=canvas_mode,
            optimization_plan=optimization_plan,
        )

        try:
            has_tools = bool(self.tool_executor)
            tools_param = OPTIMIZE_AGENT_TOOLS if has_tools else None

            if has_tools:
                # 阶段 1：非流式调用，检测是否触发工具
                shown_tools = set()  # 跟踪已展示过状态提示的工具类型，避免重复
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
                        status_msg = self._get_tool_status_message(choice.message.tool_calls, shown_tools)
                        if status_msg:
                            yield status_msg
                        # 记录已展示的工具类型
                        for tc in choice.message.tool_calls:
                            shown_tools.add(tc.function.name)
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
        canvas_mode: bool = False,
        optimization_plan: dict = None,
    ) -> str:
        """非流式生成优化建议（支持 Function Call）"""
        messages = self._build_messages(
            assessment_context, resume_text, user_message,
            conversation_summary, recent_messages or [], memory_context,
            canvas_mode=canvas_mode,
            optimization_plan=optimization_plan,
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
        canvas_mode: bool = False,
        optimization_plan: dict = None,
    ) -> list:
        """
        构建发送给 LLM 的消息列表

        结构：
        1. system prompt（canvas_mode 时追加画布指令）
        2. 上下文注入（作为第一条 user message）
        3. 历史对话（去掉当前这轮，因为当前消息单独传入）
        4. 当前用户消息

        注意：recent_messages 可能已包含当前用户消息（因为协调者先 add_message 再路由），
        所以需要排除末尾与 user_message 相同的消息，避免重复。
        """
        system_prompt = self.SYSTEM_PROMPT
        if canvas_mode:
            system_prompt += self.CANVAS_MODE_PROMPT
        messages = [{"role": "system", "content": system_prompt}]

        # 构建上下文
        context = self._build_context(ctx, resume_text, conversation_summary, memory_context, optimization_plan)

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
        conversation_summary: str, memory_context: str,
        optimization_plan: dict = None,
    ) -> str:
        """构建上下文信息"""
        factors = ctx.get("factors", {})
        abilities = ctx.get("abilities", {})

        factor_names = {
            "practical_knowledge": "知识深度",
            "managerial_knowledge": "统筹能力",
            "communication": "沟通影响",
            "thinking_environment": "问题复杂度",
            "thinking_challenge": "创新思维",
            "freedom_to_act": "决策自主性",
            "magnitude": "影响规模",
            "nature_of_impact": "贡献类型",
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
学历: {ctx.get('educationLevel', '未知')} | 专业: {ctx.get('major', '未知')}
意向城市: {ctx.get('city', '未知')} | 意向行业: {ctx.get('industry', '未知')}
企业性质: {ctx.get('companyType', '未知')} | 意向企业: {ctx.get('targetCompany', '未知')}
目标岗位: {ctx.get('jobTitle', '未知')} | 职能: {ctx.get('jobFunction', '未知')}
职级: {ctx.get('grade', '未知')} | 薪酬: {ctx.get('salaryRange', '未知')}

8维能力档位:
{factors_text}

8维能力:
{abilities_text}
</assessment>""")

        parts.append(f"<resume>\n{resume_preview}\n</resume>")

        if memory_context:
            parts.append(f"<memory>\n{memory_context}\n</memory>")

        if optimization_plan:
            try:
                plan_text = json.dumps(optimization_plan, ensure_ascii=False, indent=2)
                parts.append(f"<optimization_plan>\n{plan_text}\n</optimization_plan>")
            except (TypeError, ValueError):
                pass

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
学历: {ctx.get('educationLevel', '未知')} | 专业: {ctx.get('major', '未知')}
意向城市: {ctx.get('city', '未知')} | 意向行业: {ctx.get('industry', '未知')}
企业性质: {ctx.get('companyType', '未知')} | 意向企业: {ctx.get('targetCompany', '未知')}
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
