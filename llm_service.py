"""
===========================================
LLM 服务模块 (DeepSeek 专用版)
===========================================
支持自动验证和自纠正循环
"""

import os
import json
import time
import re
from typing import Dict, Optional, Tuple, List
from openai import OpenAI

class LLMService:
    """
    LLM 服务类 (适配 DeepSeek)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        """
        初始化 LLM 服务 - 强制使用 DeepSeek

        使用自定义httpx客户端来绕过系统代理，实现线程安全
        """
        # 优先使用传入的 Key
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')

        if not self.api_key:
            raise ValueError("未找到 API Key！请在 api_server.py 中填入你的 Key。")

        # 设置模型 (DeepSeek V3 使用 deepseek-chat)
        self.model = model

        # 🟢 使用自定义httpx客户端，绕过系统代理（线程安全方案）
        # DeepSeek API 在国内可以直接访问，不需要代理
        # 通过设置proxy=None，httpx会忽略环境变量中的代理设置
        try:
            import httpx
            # 创建不使用代理的httpx客户端
            http_client = httpx.Client(
                proxy=None,  # 显式禁用代理
                timeout=httpx.Timeout(120.0, connect=30.0)  # 连接超时30s，总超时120s
            )
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
                http_client=http_client
            )
            print("[LLM] 使用自定义httpx客户端初始化（线程安全，已禁用代理）")
        except ImportError:
            # httpx未安装时回退到旧方案（仍可工作，但非线程安全）
            print("[LLM] 警告: httpx未安装，回退到临时环境变量方案")
            from utils import temporary_env_vars
            proxy_vars_to_remove = {
                'HTTP_PROXY': None, 'HTTPS_PROXY': None,
                'http_proxy': None, 'https_proxy': None,
                'ALL_PROXY': None, 'all_proxy': None
            }
            with temporary_env_vars(proxy_vars_to_remove):
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com"
                )

    def _clean_json_content(self, content: str) -> str:
        """
        清理 DeepSeek 返回的内容，提取纯 JSON

        处理以下情况：
        1. 前面有说明文字："好的，这是结果：{...}"
        2. 包含在 Markdown 代码块中：```json{...}```
        3. 前后有空白字符或注释

        Raises:
            ValueError: 无法解析为有效JSON时抛出，包含详细错误信息
        """
        # 记录尝试过的清理方法和错误，用于最终错误报告
        attempts = []

        # 方法0：尝试直接解析（如果已经是纯JSON）
        try:
            json.loads(content.strip())
            return content.strip()
        except json.JSONDecodeError as e:
            attempts.append(f"直接解析失败: {e.msg} (位置 {e.pos})")

        # 方法1：提取 Markdown 代码块中的 JSON
        json_pattern = r'```(?:json)?\s*(.*?)\s*```'
        matches = re.findall(json_pattern, content, re.DOTALL | re.IGNORECASE)
        if matches:
            cleaned = matches[0].strip()
            try:
                json.loads(cleaned)
                return cleaned
            except json.JSONDecodeError as e:
                attempts.append(f"Markdown代码块提取失败: {e.msg}")
        else:
            attempts.append("未找到Markdown代码块")

        # 方法2：查找第一个 { 到最后一个 } 之间的内容
        brace_start = content.find('{')
        brace_end = content.rfind('}')
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            cleaned = content[brace_start:brace_end + 1]
            try:
                json.loads(cleaned)
                return cleaned
            except json.JSONDecodeError as e:
                attempts.append(f"花括号提取失败: {e.msg}")
        else:
            attempts.append("未找到有效的JSON花括号结构")

        # 方法3：移除常见的前缀文字
        prefixes_to_remove = [
            (r'^[^{]*', "移除前缀文字"),
            (r'好的[，,][^}]*?{', "移除'好的'前缀"),
            (r'根据您[的,][^}]*?{', "移除'根据您的'前缀"),
            (r'以下是[^{]*?{', "移除'以下是'前缀"),
        ]

        for prefix, desc in prefixes_to_remove:
            cleaned = re.sub(prefix, '{', content, count=1)
            try:
                json.loads(cleaned.strip())
                return cleaned.strip()
            except json.JSONDecodeError:
                pass

        attempts.append("前缀移除方法均失败")

        # 所有方法都失败了，抛出明确的错误
        content_preview = content[:200] + "..." if len(content) > 200 else content
        error_msg = (
            f"无法从LLM响应中提取有效JSON\n"
            f"尝试记录:\n  - " + "\n  - ".join(attempts) + "\n"
            f"原始内容预览:\n{content_preview}"
        )
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    def _adjust_pk_level(self, pk: str) -> str:
        """
        对 PK 档位进行 +0.5 档调整（用于补偿 chat 模型的保守倾向）

        规则：
        - E 系列及以上（E-, E, E+, F-, F, F+, G-, G, G+）：保持不变
        - E 系列以下（D+, D, D-, ...）：上调 0.5 档

        档位顺序：A- < A < A+ < B- < B < B+ < C- < C < C+ < D- < D < D+ < E- < E < E+ < F- < F < F+ < G- < G < G+
        """
        # 定义完整的档位顺序
        pk_order = [
            'A-', 'A', 'A+',
            'B-', 'B', 'B+',
            'C-', 'C', 'C+',
            'D-', 'D', 'D+',
            'E-', 'E', 'E+',
            'F-', 'F', 'F+',
            'G-', 'G', 'G+'
        ]

        # 标准化输入（处理可能的格式问题）
        pk = pk.strip().upper()
        if pk and pk[0].isalpha():
            # 确保格式正确：如 "D" -> "D", "D+" -> "D+", "D-" -> "D-"
            base = pk[0]
            suffix = pk[1:] if len(pk) > 1 else ''
            if suffix not in ['', '+', '-']:
                suffix = ''
            pk = base + suffix

        # E 系列及以上保持不变（E-, E, E+, F系列, G系列）
        high_levels = ['E-', 'E', 'E+', 'F-', 'F', 'F+', 'G-', 'G', 'G+']
        if pk in high_levels:
            return pk

        # 找到当前档位在顺序中的位置
        if pk not in pk_order:
            # 如果找不到，尝试只用基础字母
            pk_base = pk[0] if pk else 'D'
            pk = pk_base
            if pk not in pk_order:
                return 'D'  # 默认值

        current_index = pk_order.index(pk)

        # +0.5 档 = 索引 +1
        new_index = min(current_index + 1, len(pk_order) - 1)

        return pk_order[new_index]

    def extract_pk_range(
        self,
        eval_text: str,
        title: str,
        function: str = "未知",
        timeout: Optional[int] = None,
        assessment_type: str = 'CV'
    ) -> Dict:
        """
        使用LLM提取专业知识(PK)档位（带符号的单一档位）- 仅支持 CV 模式

        两步判断策略：
        1. 先判断 PK 落在哪个系列（如 E 系列、D 系列）
        2. 再判断相比标准岗位画像，该候选人的 PK 是强一点(+)、弱一点(-)、还是差不多(无符号)

        Args:
            eval_text: 简历内容
            title: 岗位名称
            function: 职能类型
            timeout: 超时时间
            assessment_type: 评估类型（仅支持 'CV'）

        Returns:
            {
                'practical_knowledge': 'E+',  # 带符号的单一档位
                'reasoning': '该候选人...'
            }
        """
        from config import config

        if timeout is None:
            timeout = config.LLM_TIMEOUT

        system_prompt = """你是HAY人才评估专家。你的任务是**根据简历内容**，评估候选人实际展现出的专业知识水平。

【最优先规则 - 输入质量判断】
如果简历内容是无意义的文字（如随机字母数字、乱码、测试文字、几个字的废话、与职业完全无关的内容），
说明无法从中提取任何有效的职业信息，此时你应该：
- 将 practical_knowledge 设为 "B"（信息不足默认档）
- 在 reasoning 中明确说明"简历内容无实质性职业信息，无法进行有效评估"
不要试图从无意义的内容中编造分析。

【重要提示 - 必须综合考虑以下因素】
- 你评估的是"候选人具备什么能力"，而不是"岗位需要什么能力"
- 必须基于简历中的具体内容来判断，重点关注：
  1. **工作年限**：总工作经验年数是判断档位的重要参考（如6-10年通常对应E档）
  2. **项目复杂度**：主导过的项目规模、影响范围、创新程度
  3. **过往雇主背景**：知名企业、头部咨询公司、500强等经历应给予加分
  4. **职位层级变化**：职业发展轨迹是否体现能力提升
- 不要根据目标岗位名称来推断，要根据简历中的实际经历来评估
- 避免保守评估：如果候选人有知名企业背景或复杂项目经验，应充分考虑

【专业知识档位说明 - 候选人能力视角】

A - 基础应用能力：能执行标准流程和基础操作
   * 典型特征：刚入职场、仅有基础培训、工作内容为简单重复性任务
   * 简历表现：实习经历为主、无独立负责的工作内容

B - 初级专业能力：经过专业培训，能独立处理常规工作
   * 典型特征：1-2年工作经验、能独立完成指定任务、需要上级指导
   * 简历表现：有1-2段正式工作经历、负责过具体执行性工作

C - 中低级专业能力：有一定经验积累，能处理较复杂的例行工作
   * 典型特征：2-4年经验、能独立解决常见问题、开始带新人
   * 简历表现：有多段相关经历、独立负责过完整项目模块、有一定专业深度

D - 中级专业能力：专业知识扎实，能分析问题并提出解决方案
   * 典型特征：4-6年经验、能处理复杂问题、有团队管理经验
   * 简历表现：主导过重要项目、有明确的业绩成果、具备跨部门协作经验

E - 中高级专业能力：深入理解专业领域，能处理复杂和非常规问题
   * 典型特征：6-10年经验、领域内资深、能制定方案和流程
   * 简历表现：有丰富的项目经验、取得过显著成果、能独立决策

F - 高级专业能力：领域专家，能创新和优化流程
   * 典型特征：10年以上经验、行业知名、能影响组织决策
   * 简历表现：有战略级项目经验、行业影响力、高管或专家头衔

G - 最高级专业能力：战略级专家，能制定行业标准或战略方向
   * 典型特征：顶级专家、行业领袖、战略决策者
   * 简历表现：有行业级影响力、制定过标准或战略、C-level经历

【两步判断法 - 基于简历内容】

第一步：判断档位系列
- 根据简历中的工作年限、职位层级、项目复杂度、成果影响力来判断
- 重点看：工作经历的深度和广度、负责过的项目规模、取得的实际成果

第二步：判断符号（精细调整）
- 在确定档位系列后，对比同档位候选人的"标准画像"：
  * 比标准更强 → 加号(+)：经历更丰富、成果更突出、能力更全面
  * 与标准相当 → 无符号：符合该档位的典型特征
  * 比标准略弱 → 减号(-)：经历稍浅、成果一般、能力有待提升

【输出要求】
1. 只返回**1个档位**（必须带符号判断：+、-、或无符号）
2. 档位格式：字母 + 可选符号，如 "E"、"E+"、"E-"
3. 推理必须引用简历中的具体内容作为依据
4. 识别候选人的最高学历及是否已毕业，输出 education 字段，取值如下：
   - "本科在读"：本科/大学尚未毕业（在读大学生）
   - "本科毕业"：已获得本科/学士学位
   - "研究生在读"：硕士尚未毕业（在读研究生）
   - "研究生毕业"：已获得硕士/研究生学位
   - "博士在读"：博士尚未毕业（在读博士生）
   - "博士毕业"：已获得博士学位
   - "未知"：简历中未提及学历信息

【输出格式】
必须严格按照以下JSON格式输出：
{
  "practical_knowledge": "D+",
  "education": "本科毕业",
  "reasoning": "第一步：根据简历显示，候选人有X年工作经验，曾在XX公司担任XX职位，主导过XX项目，属于D系列；第二步：相比标准D级候选人，该候选人在XX方面表现突出（具体引用简历内容），因此判定为D+"
}"""

        user_prompt = f"""请根据以下简历内容，评估该候选人实际展现出的专业知识水平：

【目标岗位】{title}（仅供参考，评估依据是简历内容而非岗位要求）
【所属职能】{function}

【简历内容 - 这是你评估的核心依据】
{eval_text}

请仔细阅读简历中的：
1. 工作经历：职位、公司、工作年限、具体职责
2. 项目经验：项目规模、个人角色、取得成果
3. 教育背景：学历、专业、学校
4. 技能描述：专业技能、证书资质

基于以上内容，使用两步判断法，给出该候选人的专业知识档位（如 D、D+、D-）。
注意：你的判断必须基于简历中的实际内容，而不是目标岗位的要求。"""

        try:
            print(f"[LLM] 正在调用DeepSeek提取专业知识(PK)档位...")
            print(f"[LLM调试] CV内容长度={len(eval_text) if eval_text else 0}字符")
            print(f"[LLM调试] CV内容前200字: {eval_text[:200] if eval_text else '(空)'}")

            # DeepSeek Reasoner不支持response_format参数
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,  # 确保完全确定性
                "timeout": timeout
            }

            # 只有chat模型支持response_format
            if 'reasoner' not in self.model.lower():
                api_params['response_format'] = {'type': 'json_object'}

            response = self.client.chat.completions.create(**api_params)

            content = response.choices[0].message.content
            cleaned_content = self._clean_json_content(content)
            result = json.loads(cleaned_content)

            # 验证必要字段
            if 'practical_knowledge' not in result:
                raise ValueError("LLM返回结果缺少字段: practical_knowledge")

            pk = result['practical_knowledge']
            # 如果返回了列表，取第一个
            if isinstance(pk, list):
                pk = pk[0] if pk else 'D'
            result['practical_knowledge'] = pk

            # 验证档位值
            valid_pk_levels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            pk_base = pk.rstrip('+-')
            if pk_base not in valid_pk_levels:
                raise ValueError(f"专业知识档位'{pk}'无效，必须是A-G之一")

            print(f"[LLM] ✓ 专业知识档位提取成功")
            print(f"  - 专业知识(原始): {result['practical_knowledge']}（带符号单一档位）")

            # CV 模式下，对 chat 模型结果进行 +0.5 档调整（补偿保守倾向）
            if 'reasoner' not in self.model.lower():
                original_pk = result['practical_knowledge']
                adjusted_pk = self._adjust_pk_level(original_pk)
                result['practical_knowledge'] = adjusted_pk
                result['pk_adjusted'] = True
                result['pk_original'] = original_pk
                print(f"  - 专业知识(调整后): {adjusted_pk}（CV模式+0.5档调整）")

            # 提取学历字段
            education = result.get('education', '未知')
            result['education'] = education
            print(f"  - 学历: {education}")

            if 'reasoning' in result:
                print(f"  - 推理: {result['reasoning'][:100]}...")

            return result

        except Exception as e:
            print(f"[LLM] ✗ 专业知识档位提取失败: {e}")
            raise

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        通用LLM调用方法 - 用于对话调整等场景

        参数:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            temperature: 温度参数（可选，默认使用config）
            timeout: 超时时间（可选，默认使用config）

        返回:
            str: LLM的响应文本，失败返回None
        """
        from config import config

        if temperature is None:
            temperature = config.LLM_TEMPERATURE
        if timeout is None:
            timeout = config.LLM_TIMEOUT

        try:
            print(f"[LLM] 调用DeepSeek API...")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                timeout=timeout
            )

            content = response.choices[0].message.content
            print(f"[LLM] ✓ 调用成功，响应长度: {len(content)} 字符")
            return content

        except Exception as e:
            print(f"[LLM] ✗ 调用失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def test_connection(self) -> bool:
        """
        测试DeepSeek API连接

        注：客户端已在初始化时配置为不使用代理，无需再次处理
        """
        try:
            print(f"正在测试 DeepSeek 连接...")
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            print("✓ DeepSeek 连接成功！")
            return True
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            return False

    def generate_deep_insight(
        self,
        assessment_type: str,
        city: str,
        job_title: str,
        level: int,
        level_tag: str,
        salary_range: str,
        abilities: Dict,
        original_text: str,
        timeout: Optional[int] = None,
        insufficient: bool = False,
        industry: str = '',
        job_function: str = ''
    ) -> Tuple[Optional[str], int]:
        """
        生成 AI 深度评估洞察文本（仅支持 CV 模式）

        Args:
            assessment_type: 评估类型（仅支持 "CV"）
            city: 城市
            job_title: 职位名称
            level: 职级
            level_tag: 职级标签
            salary_range: 薪酬区间
            abilities: 五维能力数据 {"专业力": {"score": 70, "explanation": "..."}, ...}
            original_text: 简历内容
            timeout: 超时时间
            insufficient: 是否信息不足模式（使用赛道分析 prompt）
            industry: 行业
            job_function: 职能

        Returns:
            (深度洞察文本, 简历健康度分数) 元组。文本失败返回 None，分数范围 1-100
        """
        from config import config
        if timeout is None:
            timeout = config.LLM_TIMEOUT

        print(f"[LLM] 开始生成 AI 深度评估...{'（信息不足-赛道分析模式）' if insufficient else ''}")

        # 信息不足模式：直接返回固定低分
        if insufficient:
            system_prompt = self._build_insufficient_deep_insight_prompt(
                city, industry, job_function, job_title
            )
        else:
            system_prompt = self._build_deep_insight_system_prompt(
                assessment_type, city, job_title, level, level_tag,
                salary_range, abilities, original_text
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "请基于以上信息，生成深度洞察分析。"}
                ],
                temperature=0.3,
                timeout=timeout
            )

            content = response.choices[0].message.content.strip()

            # 提取简历健康度分数
            resume_health_score = 25  # 默认低分
            if insufficient:
                resume_health_score = 25
            else:
                health_match = re.search(r'\[RESUME_HEALTH_SCORE:(\d+)\]', content)
                if health_match:
                    resume_health_score = max(1, min(100, int(health_match.group(1))))
                    # 从正文中移除标签
                    content = re.sub(r'\s*\[RESUME_HEALTH_SCORE:\d+\]\s*', '', content).strip()
                    print(f"[LLM] 简历健康度分数: {resume_health_score}")
                else:
                    resume_health_score = 50  # 未输出标签时给中间值
                    print(f"[LLM] ⚠️ 未检测到简历健康度标签，使用默认值: {resume_health_score}")

            # 检查字数
            char_count = len(content)
            print(f"[LLM] 深度评估字数: {char_count}字")

            if char_count < 30:
                print(f"[LLM] ⚠️ 深度评估文字过短，可能生成失败")
                return None, resume_health_score

            print(f"[LLM] ✓ AI 深度评估生成完成")
            return content, resume_health_score

        except Exception as e:
            print(f"[LLM] ✗ AI 深度评估生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None, 25

    def _build_insufficient_deep_insight_prompt(
        self,
        city: str,
        industry: str,
        job_function: str,
        job_title: str
    ) -> str:
        """构建信息不足时的赛道分析 Prompt（基于城市+行业+职能+岗位）"""

        return f"""你是一位资深的职业市场分析师和薪酬顾问，拥有丰富的人力资源和猎头经验。

## 背景
用户提交了一份信息较少的简历，无法进行个人维度的精准分析。但我们已知用户的基本定位信息，请基于这些信息生成一份赛道维度的市场洞察。

## 已知信息
- 城市：{city}
- 行业：{industry}
- 职能：{job_function}
- 目标岗位：{job_title}

## 输出框架（严格按以下结构输出，段落之间空一行）

**固定开场白（必须原文输出，不得修改）**：由于您目前提供的信息较为简略，暂时无法为您生成个性化的职业价值分析结果。

第一段（2-3句话）：话锋一转，说基于城市、行业和岗位信息，可以从赛道视角帮您看清大局。语气温暖自然。

第二段（4-5句话）：基于【{city} + {industry}】的赛道生存现状分析。该城市该行业目前处于什么阶段（红利期/稳步扩张/内卷加剧/转型阵痛），人才供需关系如何，薪酬水平在全国同行业中的定位。要有具体洞察，不要泛泛而谈。

第三段（5-6句话）：基于【{job_function} + {job_title}】的职业跃迁分析。该岗位的典型晋升通路（用【】标注关键节点，如【执行者→模块负责人→业务线负责人】），薪酬跃迁的关键卡点，当前市场最看重的核心能力，以及1-2个可操作的提升建议。

**固定结尾（必须原文输出，不得修改，单独成段）**：若您希望获得个性化的价值和能力解析报告，可以在新一轮的测评中提供更详细的信息哦（如具体工作经历、关键项目经历及成果、专业技能与优势能力等）。您提供的信息越完整、越真实，我们的分析模型也将能够给出更精准的判断与建议。

## 输出要求

1. **总字数**：400-550字
2. **结构**：严格3段，段落之间空一行
3. **语气**：专业但不端着，通俗易懂
4. **关键词高亮**：每段中最重要的2-3个关键词或短语用 **双星号** 包裹（如 **红利期**、**薪资天花板**）
5. **晋升节点**：用【】标注（如【初级→中级】）
6. **绝对禁止**：
   - 禁止输出"模块一"、"模块二"、"模块三"或任何段落标题/小标题文字
   - 禁止以"您好"、"看到您提交的信息了"等寒暄/问候开头
   - 禁止使用 markdown 标记（除双星号高亮外）、序号或 bullet points
   - 禁止提及"职级"、"等级"、"level"、具体分数
   - 禁止编造用户的个人经历
   - 禁止描述任何具体的薪酬现象或问题，如"薪酬倒挂"、"薪资倒挂"、"薪资压缩"、"薪酬泡沫"等负面表述；可以描述该行业整体薪酬水平高的正向信息
   - 直接输出正文内容，不要有任何前缀、标题、问候语"""

    def _build_deep_insight_system_prompt(
        self,
        assessment_type: str,
        city: str,
        job_title: str,
        level: int,
        level_tag: str,
        salary_range: str,
        abilities: Dict,
        original_text: str
    ) -> str:
        """构建深度洞察的 System Prompt"""

        # 提取能力分数和解释
        def get_ability_info(name: str) -> Tuple[float, str]:
            info = abilities.get(name, {})
            score = info.get('score', 50) / 10  # 百分制转十分制
            explanation = info.get('explanation', '暂无数据')
            return score, explanation

        pro_score, pro_exp = get_ability_info('专业力')
        mgmt_score, mgmt_exp = get_ability_info('管理力')
        collab_score, collab_exp = get_ability_info('合作力')
        think_score, think_exp = get_ability_info('思辨力')
        innov_score, innov_exp = get_ability_info('创新力')

        # 限制原始文本长度
        text_preview = original_text[:2000] if len(original_text) > 2000 else original_text

        type_label = "CV 模式（个人简历评估）"
        input_label = "简历"

        prompt = f"""你是一位资深的职业价值评估专家和薪酬谈判顾问，拥有丰富的人力资源和猎头经验。

你的任务是基于用户的评估结果，生成一段个性化的深度洞察分析，核心目标是解释「为什么你值这个薪酬」。

## 最重要的前置规则：输入质量判断

在生成任何分析之前，你必须先判断用户提供的简历/履历内容是否包含足够的、有意义的职业信息。

**如果原始简历内容存在以下任何一种情况，你必须拒绝编造分析，直接输出诚实的反馈：**
- 内容是随机的字母、数字、符号或乱码，没有任何实际含义（如"abc123"、"asdfgh"、"111222"）
- 内容过于简短（只有几个字或一两句没有实质内容的话），根本无法提取任何有价值的职业信息
- 内容明显是胡编乱造的废话、测试文字、或与职业经历完全无关的内容
- 内容虽然有一些文字，但完全缺乏具体的工作经历、项目经验、技能描述等关键信息

**当输入不合格时，你的输出必须是（直接输出以下内容，不要按后面的4段框架）：**

很抱歉，根据你提供的内容，我**无法进行有效的职业价值分析**。

目前提供的信息太少或者不包含有意义的职业经历描述，系统没有办法据此评估你的市场价值。一份有效的分析至少需要了解你的**工作经历**（在哪些公司做过什么岗位）、**项目经验**（负责过什么项目、取得了什么成果）、以及**专业技能**等基本信息。

建议你重新填写，尽量详细地描述你的**真实职业经历和核心成果**，哪怕是用几段话概括也好——只要包含真实的工作信息，系统就能给出有价值的分析和薪酬定位。内容越详细、越真实，分析结果就越准确。

**只有当简历内容确实包含有意义的、可分析的职业信息时，才按照下面的「分析框架」正常输出4段分析。**

## 评估类型：{type_label}

## 输入信息
- 城市：{city}
- 目标职位：{job_title}
- 月度基本工资区间：{salary_range}
- 五维能力得分（满分10分）：
  - 专业力：{pro_score:.1f}分 - {pro_exp}
  - 管理力：{mgmt_score:.1f}分 - {mgmt_exp}
  - 合作力：{collab_score:.1f}分 - {collab_exp}
  - 思辨力：{think_score:.1f}分 - {think_exp}
  - 创新力：{innov_score:.1f}分 - {innov_exp}

## 原始{input_label}内容
{text_preview}

## 分析框架（仅在输入内容合格时使用）
"""

        prompt += """
### 分析要点（按以下顺序输出4个段落）：

**第一段：定价总结**（3-4句话）
- 开头必须是："你的月度基本工资定位在 {月度基本工资区间}，"（注意：这里是月base，不是年薪）
- 用通俗的语言概括为什么值这个价
- 简要说明你的经历在市场上处于什么水平

**第二段：你的核心竞争力**（5-6句话）
- 从简历中提取 2-3 个最亮眼的经历（引用原文）
- 用自然语言说明这些经历为什么有市场价值
- 解释这些经历体现了什么能力，为什么企业愿意为此付费
- 重点是「这些经历能帮你在面试中脱颖而出」

**第三段：可以更值钱的地方**（4-5句话）
- 基于简历内容，指出哪些方面如果加强，薪酬可以更高
- 给出 1-2 个具体的努力方向
- 说明达到什么程度可以突破到更高薪酬区间
- 语气要鼓励而非批评，是"提升空间"而非"缺点"

**第四段：面试亮点建议**（3-4句话）
- 面试时可以重点讲的 1-2 个故事/经历
- 具体怎么讲才能突出价值
- 可能被追问的问题，以及如何巧妙回应
"""

        prompt += """
## 输出要求

1. **总字数**：400-500字
2. **结构**：严格按照上述4个段落输出，每个段落单独成段，段落之间空一行，便于阅读
3. **引用原文**：关键段落引用 1-2 处简历原文（用引号标注）
4. **语气**：像朋友在帮你分析，专业但不端着，通俗易懂
5. **关键词高亮**：对每段中最重要的2-3个关键词或短语，用 **双星号** 包裹（如 **核心竞争力**、**薪资溢价空间**），便于前端高亮展示
6. **禁止**：
   - 除了上述关键词高亮外，不要使用其他 markdown 标记、序号或 bullet points
   - 不要提及"职级"、"等级"、"level"、具体分数（如"5.7分"）等
   - 不要用"专业力""管理力"等术语，用日常语言描述能力
   - 不要泛泛而谈，每句话都要有具体依据

## 简历健康度评分

正文输出完毕后，最后单独一行输出：[RESUME_HEALTH_SCORE:XX]（XX为1-100整数，综合评估简历的信息完整度、内容深度和量化成果）。
"""

        return prompt
