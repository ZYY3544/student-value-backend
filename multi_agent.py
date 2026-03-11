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
from utils import safe_json_parse


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

    SYSTEM_PROMPT = """你是 Sparky，用户的专属求职伙伴。请生成一段个性化的欢迎语。

【你的输入】
用户的评测结果（含5维能力得分）和简历内容。

【欢迎语要求】
- 60-120字，轻松简洁，像一个靠谱的小助手
- 点出1个用户的亮点或特色（基于评测数据或简历内容，具体到维度或经历）
- 简要说明你能帮做的事（简历优化、JD定制改写、岗位搜索等），不要逐条列举，用自然语言带过
- 以一个开放式问题收尾，引导用户开始
- 禁止emoji、禁止提及HAY/八因素等方法论术语
- 禁止分析简历细节，只点到为止
- 不要用 markdown 标题"""

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
            import traceback
            print(f"[DiagnosisAgent] 诊断生成失败: {e}")
            traceback.print_exc()
            # 兜底：基于评测数据拼一个还算个性化的开场白
            job_title = assessment_context.get("jobTitle", "")
            title_part = f"，看到你的目标是**{job_title}**方向" if job_title else ""
            return (
                f"嗨，我是 Sparky！已经看过你的评测结果和简历了{title_part}。"
                "我可以帮你优化简历、按目标 JD 定制改写、搜索岗位信息，你想先从哪里开始？"
            )

    def _build_input(self, ctx: dict, resume_text: str) -> str:
        """构建诊断 Agent 的输入"""
        factors = ctx.get("factors", {})
        abilities = ctx.get("abilities", {})
        grade = ctx.get("grade", "未知")
        salary = ctx.get("salaryRange", "未知")
        job_title = ctx.get("jobTitle", "未知")
        job_function = ctx.get("jobFunction", "未知")

        factor_names = {
            "practical_knowledge": "专业力",
            "managerial_knowledge": "管理力",
            "communication": "合作力",
            "thinking_environment": "思辨力",
            "thinking_challenge": "创新力",
            "freedom_to_act": "管理力(FTA)",
            "magnitude": "创新力(M)",
            "nature_of_impact": "合作力(NI)",
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

        # 找出偏弱的能力维度
        weak_abilities = []
        strong_abilities = []
        if isinstance(abilities, dict):
            sorted_abs = sorted(abilities.items(), key=lambda x: x[1].get("score", 50))
            weak_abilities = [name for name, _ in sorted_abs[:2]]
            strong_abilities = [name for name, _ in sorted_abs[-2:]]

        return f"""请对以下用户的评估结果和简历进行诊断分析。

【评估结果】
- 学历: {ctx.get('educationLevel', '未知')} | 专业: {ctx.get('major', '未知')}
- 意向城市: {ctx.get('city', '未知')} | 意向行业: {ctx.get('industry', '未知')}
- 企业性质: {ctx.get('companyType', '未知')} | 意向企业: {ctx.get('targetCompany', '未知')}
- 目标岗位: {job_title}
- 所属职能: {job_function}
- 薪酬区间: {salary}

5维能力档位:
{factors_text}

5维能力得分:
{abilities_text}

能力分析提示:
- 较强维度: {', '.join(strong_abilities) if strong_abilities else '暂无'}
- 偏弱维度: {', '.join(weak_abilities) if weak_abilities else '暂无'}
（注意：应届生管理力偏弱属于正常情况，诊断时不必过度强调）

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
    - 分析评测结果中的偏弱维度
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
- 能力体现：评测中偏弱的能力维度，在简历哪些段落可以加强体现
- 差异化：哪些经历有独特价值但没有突出
- 求职阶段：用户是在探索方向、准备材料、还是已经在投递

【主动执行建议规则】
- 如果用户目标岗位明确 → 建议 search_jobs 搜索相关岗位了解市场
- 如果用户评测结果中某个能力维度特别突出 → 在 career_insight 中点出来
- 如果用户意向行业/方向比较热门 → 建议看看竞争态势
- proactive_actions 给出 1-3 个建议

【规则】
- 优先关注评测得分偏弱的能力维度对应的简历段落
- 注意：应届生的管理力偏弱是正常的（缺少团队管理经验），在诊断和建议中不要过度强调，可以一笔带过
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
            plan = safe_json_parse(content)
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
            "practical_knowledge": "专业力",
            "managerial_knowledge": "管理力",
            "communication": "合作力",
            "thinking_environment": "思辨力",
            "thinking_challenge": "创新力",
            "freedom_to_act": "管理力(FTA)",
            "magnitude": "创新力(M)",
            "nature_of_impact": "合作力(NI)",
        }

        factors_text = "\n".join(
            f"  - {factor_names.get(k, k)}: {v}"
            for k, v in factors.items()
        )

        abilities_text = "\n".join(
            f"  - {name}: {info.get('score', '?')}分 ({info.get('level', '?')})"
            for name, info in abilities.items()
        ) if isinstance(abilities, dict) else "  暂无能力数据"

        # 能力排序，找出偏弱维度
        weak_abilities = []
        strong_abilities = []
        if isinstance(abilities, dict):
            sorted_abs = sorted(abilities.items(), key=lambda x: x[1].get("score", 50))
            weak_abilities = [name for name, _ in sorted_abs[:2]]
            strong_abilities = [name for name, _ in sorted_abs[-2:]]

        resume_preview = resume_text[:3000] if len(resume_text) > 3000 else resume_text

        return f"""请分析以下评估结果和简历，生成优化计划。

【评估结果】
- 学历: {ctx.get('educationLevel', '未知')} | 专业: {ctx.get('major', '未知')}
- 意向城市: {ctx.get('city', '未知')} | 意向行业: {ctx.get('industry', '未知')}
- 目标岗位: {ctx.get('jobTitle', '未知')}
- 薪酬区间: {ctx.get('salaryRange', '未知')}

5维能力档位:
{factors_text}

5维能力得分:
{abilities_text}

能力分析:
- 较强维度: {', '.join(strong_abilities) if strong_abilities else '暂无'}
- 偏弱维度: {', '.join(weak_abilities) if weak_abilities else '暂无'}
（注意：应届生管理力偏弱是正常情况，不必过度强调）

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

    SYSTEM_PROMPT = """你是一位经验丰富的简历优化顾问，擅长将平淡的简历描述改写为有说服力的、体现候选人真实价值的文字。

【格式规则】绝对禁止使用任何 emoji 或表情符号。列表用数字序号或短横线。

【改写方法（内部参考，不向用户展示方法论名称）】

1. 背景-任务-行动-成果结构强化
2. 量化数据注入：能量化的必须量化，没精确数字用范围（「数十个」「百万级」），用户没提供则追问，绝不编造
3. 动词升级：执行→负责/主导，参与→核心参与，帮助→推动/促进
4. 能力维度驱动改写（核心闭环）：
   上下文中有用户5维能力评分，你必须利用这些数据指导改写方向：

   | 维度 | 改写方向 | 关键词 |
   |---|---|---|
   | 专业力 | 专业知识、行业认知、技术栈深度 | 掌握/熟练运用/深入研究 |
   | 管理力 | 项目管理、资源协调、独立决策 | 统筹/规划/协调N个部门/带领N人团队/独立负责 |
   | 合作力 | 跨部门协作、汇报沟通、影响决策 | 推动/促成/与XX部门协作达成/核心负责人 |
   | 思辨力 | 问题复杂度、挑战性、约束条件 | 强调背景复杂度/分析/洞察 |
   | 创新力 | 新方法、新思路、流程优化 | 首次/创新性地/从0到1搭建 |

   闭环要求：找出得分偏弱的2个维度 → 定位相关简历段落 → 改写时告知用户提升了哪个维度 → 每条改写关联具体维度。用户未指定改哪里时，优先推荐偏弱维度对应段落。注意：应届生管理力偏弱属于正常情况，不必过度关注。

5. 红线：禁止编造经历、编造数字、改变事实。需要数据但用户没提供时用「[待补充: 具体数字]」标记。如果用户要求编造不存在的经历，礼貌拒绝。

【对话节奏】
- 用户闲聊或打招呼时简短回应，不要主动展开简历分析
- 用户表示同意/确认时，直接执行上轮建议，不要重复追问

【输出格式（仅在改写简历时使用）】
- 一次聚焦 1-2 个点，不要一口气给太多
- 先简短引用原文，然后**必须给出改写后的完整文本**（不能只分析不改写）
- 用通俗语言说明改写提升了哪个能力维度
- 改完一个点后，根据当前对话进展自然地推荐下一步

【工具能力】
你拥有以下工具，可以在对话中主动调用：
- search_jobs：搜索市场岗位信息。仅当用户**明确要求搜索岗位**时使用，用户说"帮我改简历"时不要调这个。
- re_evaluate_resume：重新评估优化后的简历，生成新的能力评分并与之前做对比。
- compare_with_jd：将简历与 JD 做匹配分析。
- tailor_resume_to_jd：一站式 JD 定制改简历（核心能力）。解析 JD → 比对差距 → 直接输出定制化改写。当用户提供了 JD 时优先用这个。
- fetch_url：抓取网页内容，获取完整 JD 或招聘链接。
- verify_job_posting：识别岗位信息是否可能是虚假招聘。
- compare_multiple_jds：横向对比多个 JD，推荐最适合的岗位。
- save_resume_version / list_resume_versions / switch_resume_version：简历多版本管理。
调用工具前先用一句自然语言告知用户你在做什么，不要静默调用。

【核心产品链路】

1. JD 定制改写（核心能力，主动推动）：
   用户提供 JD（文本/链接/搜索结果）→ 如果是链接先 fetch_url → 调 tailor_resume_to_jd 直接输出定制化改写 → 确认后建议 save_resume_version。
   不要只分析不改写，用户的期望是"给你 JD，你直接帮我改好"。
   链路中途不要停下来重复上一步。用户说"好/继续/可以"时接着执行下一步，不要重新搜索。

2. 重评估闭环：
   改写了 2-3 个段落且用户满意时，主动建议重新评估。将改过的段落替换回完整简历，调 re_evaluate_resume，用自然语言展示能力分数和薪酬变化。
   不要编造对比数据，以工具返回的真实结果为准。禁止展示职级编号变化。

3. 简历导出：
   优化完成或用户表示"改完了"时，提醒可以在画布右上角导出 PDF/Word。投递用 PDF，编辑用 Word。

4. 多版本管理：
   用户提到不同投递方向时，告知可以针对不同公司维护不同版本。定制改写完成后主动建议保存版本。

【信息安全】
- search_jobs 返回 trust_level="低" 或 warnings 时，必须主动警告用户具体风险
- 可疑 JD 主动调 verify_job_posting 鉴别

【搜索结果展示】
- 列出每条结果的标题和实际URL（用工具返回的 link 字段），让用户可点击查看原文
- 搜索为空时如实告知，建议用户直接发 JD 给你或换关键词
- source 为 "llm_knowledge" 时说明这是已有信息，建议用户去招聘网站确认

【模拟面试】
用户选择模拟面试时，先确认目标岗位和面试类型，然后每次只问1个问题，等用户回答后给出具体反馈和示范回答。基于简历内容和目标岗位设计问题，语气像真正的面试官：专业但友善。

【方向引导】
- 用户表达迷茫时，先用引导式提问帮他理清方向，不要急着改简历
- 结合评测数据帮学生理解自己的能力画像和适合的方向

【方法论保密】
- 禁止提及"HAY"、"八因素"、"KH"、"PS"、"ACC"、"职级"、"等级"、"level"、"grade"等术语
- 用户追问时自然回应"我们从专业能力、业务影响等多个维度综合评估"

【城市信息】
所有涉及城市的操作必须使用评测上下文中的意向城市，不要自行假设。

【语气风格】
像一个靠谱的小助手，简洁干练、口语化、鼓励为主。

【优化计划引导】
用户没指定改哪里时，如果上下文中有优化计划，参考其优先级建议主动引导。用自然的对话语气，不要暴露"优化计划"这个概念。"""

    CANVAS_MODE_PROMPT = """

【Canvas模式 — 简历编辑指令（最高优先级）】
你现在处于简历画布模式。用户可以在右侧面板实时看到简历全文。

**核心规则：你必须直接输出改写后的具体文本，不能只做分析。**

每条修改必须用以下格式输出（用户点"接受"后系统会自动替换简历原文）：

<<<EDIT
SECTION: {段落标题}
ORIGINAL: {原文片段（从简历中精确复制）}
SUGGESTED: {改写后的完整片段（这是你的改写成果，必须是可直接替换原文的完整文本）}
RATIONALE: {1句话修改理由，必须说明提升了哪个能力维度，如"提升管理力，通过量化团队规模和协调成果"}
EDIT>>>

【Canvas模式 — 行为准则】
- 用户让你改什么，你就直接改，输出 EDIT 指令。不要只分析不改写、不要让用户再确认一次才给改写结果。
- 回复格式：1-2句简短说明 + EDIT 指令。不要写长篇分析。
- SUGGESTED 字段必须包含改写后的完整文本，不能是空的或只有分析说明。
- 不要在改完后追问"要不要继续看看其他部分"。
- 只有用户主动要求时，才分析或修改其他内容。"""

    TEMPERATURE = 0.4

    # 工具调用最大循环次数（防止无限调用）
    # 提升到 5 轮以支持多步链路：如 search→fetch_url→tailor_resume_to_jd→save_version
    MAX_TOOL_ROUNDS = 5

    def __init__(self, client: OpenAI, model: str, tool_executor=None):
        self.client = client
        self.model = model
        self.tool_executor = tool_executor

    def _get_tool_status_message(self, tool_calls, shown_tools: set = None) -> str:
        """
        工具执行期间的过渡提示（LLM 已在调用前用自然语言说明了意图，这里只做轻量过渡）
        """
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
        canvas_mode: bool = False,
        optimization_plan: dict = None,
    ) -> Generator[str, None, None]:
        """
        流式生成优化建议（支持 Function Call）

        策略：始终用流式调用。流中检测 tool_calls：
        - 如果检测到工具调用 → 执行工具 → 再次流式调用输出最终回复
        - 如果没有工具调用 → 直接流式输出（省掉一次 LLM 调用）

        相比之前的 probe + stream 双调用，每轮无工具场景省掉 ~50% token。

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
                shown_tools = set()

                for round_idx in range(self.MAX_TOOL_ROUNDS):
                    # 单次流式调用（带工具），同时处理 content 和 tool_calls
                    # LLM 中 tool_calls 和 content 互斥：
                    #   - 模型决定调工具 → 流中只有 tool_calls delta，无 content
                    #   - 模型直接回复 → 流中只有 content delta，无 tool_calls
                    # 因此可以安全地边流式输出 content，边收集 tool_calls
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.TEMPERATURE,
                        tools=tools_param,
                        stream=True,
                    )

                    tool_calls_map = {}  # index -> {id, function: {name, arguments}}
                    has_content = False
                    finish_reason = None

                    for chunk in stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        finish_reason = chunk.choices[0].finish_reason or finish_reason

                        # 流式输出文本内容（不触发工具时）
                        if delta.content:
                            has_content = True
                            yield delta.content

                        # 收集 tool_calls（触发工具时，content 为空）
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                if idx not in tool_calls_map:
                                    tool_calls_map[idx] = {
                                        "id": "",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                if tc_delta.id:
                                    tool_calls_map[idx]["id"] = tc_delta.id
                                if tc_delta.function:
                                    if tc_delta.function.name:
                                        tool_calls_map[idx]["function"]["name"] += tc_delta.function.name
                                    if tc_delta.function.arguments:
                                        tool_calls_map[idx]["function"]["arguments"] += tc_delta.function.arguments

                    # 流结束后判断走向
                    if tool_calls_map and finish_reason == "tool_calls":
                        # 工具调用路径
                        print(f"[OptimizeAgent] 第 {round_idx + 1} 轮工具调用（流式检测）")

                        # 构造兼容对象
                        class _ToolCall:
                            def __init__(self, tc_dict):
                                self.id = tc_dict["id"]
                                self.function = type('F', (), {
                                    'name': tc_dict["function"]["name"],
                                    'arguments': tc_dict["function"]["arguments"],
                                })()

                        tc_objects = [_ToolCall(tool_calls_map[i]) for i in sorted(tool_calls_map.keys())]

                        status_msg = self._get_tool_status_message(tc_objects, shown_tools)
                        if status_msg:
                            for ch in status_msg:
                                yield ch
                        for tc in tc_objects:
                            shown_tools.add(tc.function.name)
                        messages = self._execute_tool_calls(tc_objects, messages)
                        continue
                    else:
                        # 纯文本回复已在上面 yield 完毕，直接返回
                        print(f"[OptimizeAgent] 流式直出完成（省掉 probe 调用）")
                        return

                # 超过最大工具轮次，最终流式输出（不带工具）
                print(f"[OptimizeAgent] 超过最大工具轮次，最终流式输出")

            # 无工具场景 / 工具轮次耗尽后的兜底：纯流式输出
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
            "practical_knowledge": "专业力",
            "managerial_knowledge": "管理力",
            "communication": "合作力",
            "thinking_environment": "思辨力",
            "thinking_challenge": "创新力",
            "freedom_to_act": "管理力(FTA)",
            "magnitude": "创新力(M)",
            "nature_of_impact": "合作力(NI)",
        }

        factors_text = "\n".join(
            f"  - {factor_names.get(k, k)}: {v}"
            for k, v in factors.items()
        )

        if isinstance(abilities, dict) and abilities:
            # 按分数排序，标注偏弱维度
            sorted_abilities = sorted(
                abilities.items(),
                key=lambda x: x[1].get('score', 0) if isinstance(x[1], dict) else 0
            )
            abilities_lines = []
            for i, (name, info) in enumerate(sorted_abilities):
                if isinstance(info, dict):
                    score = info.get('score', '?')
                    level = info.get('level', '?')
                    weak_tag = " ⚠ 偏弱维度" if i < 3 else ""
                    abilities_lines.append(f"  - {name}: {score}分 ({level}){weak_tag}")
                else:
                    abilities_lines.append(f"  - {name}: {info}")
            abilities_text = "\n".join(abilities_lines)
        else:
            abilities_text = "  暂无"

        resume_preview = resume_text[:3000] if len(resume_text) > 3000 else resume_text

        parts = []

        if conversation_summary:
            parts.append(f"=== 对话摘要 ===\n{conversation_summary}\n=== /对话摘要 ===")

        parts.append(f"""=== 评测结果 ===
学历: {ctx.get('educationLevel', '未知')} | 专业: {ctx.get('major', '未知')}
意向城市: {ctx.get('city', '未知')} | 意向行业: {ctx.get('industry', '未知')}
企业性质: {ctx.get('companyType', '未知')} | 意向企业: {ctx.get('targetCompany', '未知')}
目标岗位: {ctx.get('jobTitle', '未知')} | 职能: {ctx.get('jobFunction', '未知')}
薪酬: {ctx.get('salaryRange', '未知')}

5维能力档位:
{factors_text}

5维能力:
{abilities_text}
=== /评测结果 ===""")

        parts.append(f"=== 简历原文 ===\n{resume_preview}\n=== /简历原文 ===")

        if memory_context:
            parts.append(f"=== 记忆上下文 ===\n{memory_context}\n=== /记忆上下文 ===")

        if optimization_plan:
            try:
                plan_text = json.dumps(optimization_plan, ensure_ascii=False, indent=2)
                parts.append(f"=== 优化计划 ===\n{plan_text}\n=== /优化计划 ===")
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

    SYSTEM_PROMPT = """你是一位亲切的简历优化顾问，负责在对话结束时做轻松的小总结。

【要求】
- 简单列出本次改了哪几个地方，用通俗语言说说每个改动让简历哪方面更强
- 给 1-2 个下次可以继续打磨的方向
- 正能量收尾
- 150-250字，像朋友聊天，不像写报告
- 禁止emoji、禁止提及HAY/八因素/职级等底层术语
- 用**加粗**高亮关键词，不用 markdown 标题"""

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
薪酬区间: {ctx.get('salaryRange', '未知')}""")

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
