"""
===========================================
LLM 服务模块
===========================================
支持 AWS Bedrock Sonnet / GLM 等多种后端
通过 ModelRouter 自动选择模型
"""

import os
import json
import time
import re
from typing import Dict, Optional, Tuple, List
from openai import OpenAI

class LLMService:
    """
    LLM 服务类

    初始化时传入已创建的 client（如 GLMCompatibleClient / BedrockOpenAIClient）
    """

    def __init__(self, client, model: str = "glm-4.5"):
        """
        初始化 LLM 服务

        Args:
            client: 外部客户端（如 GLMCompatibleClient / BedrockOpenAIClient）
            model: 模型名称
        """
        self.model = model

        if client is None:
            raise ValueError("必须传入 LLM client，请检查 GLM 或 Sonnet 配置。")

        self.client = client
        self.api_key = None
        print(f"[LLM] 使用外部客户端初始化 (model={model})")

    def _clean_json_content(self, content: str) -> str:
        """
        清理 LLM 返回的内容，提取纯 JSON

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
        对 PK 档位进行 +0.5 档调整（补偿 DeepSeek chat 模型的保守倾向）

        E 系列及以上保持不变，E 以下上调 0.5 档（索引 +1）
        """
        pk_order = [
            'A-', 'A', 'A+',
            'B-', 'B', 'B+',
            'C-', 'C', 'C+',
            'D-', 'D', 'D+',
            'E-', 'E', 'E+',
            'F-', 'F', 'F+',
            'G-', 'G', 'G+'
        ]

        pk = pk.strip().upper()
        if pk and pk[0].isalpha():
            base = pk[0]
            suffix = pk[1:] if len(pk) > 1 else ''
            if suffix not in ['', '+', '-']:
                suffix = ''
            pk = base + suffix

        # E 系列及以上保持不变
        high_levels = ['E-', 'E', 'E+', 'F-', 'F', 'F+', 'G-', 'G', 'G+']
        if pk in high_levels:
            return pk

        if pk not in pk_order:
            pk_base = pk[0] if pk else 'D'
            pk = pk_base
            if pk not in pk_order:
                return 'D'

        current_index = pk_order.index(pk)
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

A - 基础应用能力：能执行标准流程和基础操作；仅有基础培训就可以上岗工作
   * 典型人群：一般是保洁、保安等最底层最基础的工作，通常这类人群没有简历

B - 初级专业能力：经过专业培训，能从事略微有技术含量的工作；需要一定时间（通常是半个月左右）的培训才能上岗，甚至部分工种需要有技能证书认定
   * 典型人群：通常是工厂里的蓝领技术工人，这类人通常也不会有简历

C - 中低级专业能力：经过系统性的培训，能从事技术含量更高的工作，是工厂里的高级蓝领技术工人，需要接受的培训时间更长，更系统，与B的区别在于C的蓝领工人工作经验更丰富，工作时间更长，技能更高；或者接受过完整的高等教育（通常是大专及以上学历），具备现代职场工作素养
   * 典型人群：一类是高级蓝领技术工人；另一类是尚未毕业的大学生
   * 【硬性规则】本科在读且未毕业 → 必须给C-；硕士在读且未毕业 → 必须给C。即使有优秀的实习经历或项目经验，在读未毕业学生的PK也不得超过C，因为尚未完成学业意味着专业知识体系不完整。

D - 中级专业能力：经过系统性的培训，并且具备一定的带蓝领工人团队的经验，是工厂里的具有资深技术经验的主任或者主管角色；或者接受过完整的高等教育（通常是大学及以上），专业知识扎实，能分析问题并提出解决方案
   * 典型人群：D是蓝领工人的终点，也是白领（主要是毕业生）的起点：一类是工厂里面能够管理成百上千规模的主任或者主管；另一类是已经毕业的大学生，通常他们的工作经验在1-3年，如果靠近3年或者略微超过3年，可以优先考虑D+，如果刚毕业，可以优先考虑D-
   * 【硬性规则】D档的前提是已经毕业，在读学生不得给D或更高档位
 
E - 中高级专业能力：深入理解专业领域，能处理复杂和非常规问题
   * 典型人群：3-8年工作经验，如果工作经验较浅，可以考虑E-或者D+，如果工作经验较深且项目经历很丰富，可以考虑E+，通常工作5年及以上，可以给到E；
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
            print(f"[LLM] 正在调用 {self.model} 提取专业知识(PK)档位...")
            print(f"[LLM调试] CV内容长度={len(eval_text) if eval_text else 0}字符")
            print(f"[LLM调试] CV内容前200字: {eval_text[:200] if eval_text else '(空)'}")

            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,  # 确保完全确定性
                "timeout": timeout
            }

            # DeepSeek chat 模型支持 response_format
            is_deepseek = 'deepseek' in self.model.lower()
            is_reasoner = 'reasoner' in self.model.lower()
            if is_deepseek and not is_reasoner:
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

            # DeepSeek chat +0.5 档补偿已移除，直接使用 LLM 原始输出

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
            print(f"[LLM] 调用 {self.model} API...")

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
        """测试 LLM API 连接"""
        try:
            print(f"正在测试 {self.model} 连接...")
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            print(f"✓ {self.model} 连接成功！")
            return True
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            return False
