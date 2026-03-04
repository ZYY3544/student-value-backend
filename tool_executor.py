"""
===========================================
Function Call 工具执行器 (Tool Executor)
===========================================
为 OptimizeAgent 提供外部工具调用能力：
1. search_jobs — 搜索招聘岗位
2. re_evaluate_resume — 用 HAY 体系重新评估优化后简历
3. compare_with_jd — LLM 对比简历与 JD 匹配度
"""

import json
from typing import Optional


class ToolExecutor:
    """
    工具执行器 — 路由并执行 OptimizeAgent 的 Function Call

    依赖：
    - llm_service: LLMService 实例（用于 LLM 调用）
    - convergence_engine: IncrementalConvergence 实例（用于重新评估）
    - conversation_memory: ConversationMemory 实例（获取上下文）
    """

    def __init__(self, llm_service, convergence_engine, conversation_memory):
        self.llm_service = llm_service
        self.convergence_engine = convergence_engine
        self.conversation_memory = conversation_memory
        # 缓存原始评测结果，用于重评估对比
        self._original_assessment = None

    def set_original_assessment(self, assessment_context: dict):
        """缓存原始评测结果，供 re_evaluate_resume 做对比"""
        self._original_assessment = assessment_context

    def execute(self, tool_name: str, arguments: dict) -> str:
        """
        路由到对应工具函数，返回 JSON 字符串结果

        Args:
            tool_name: 工具名称
            arguments: 工具参数字典

        Returns:
            JSON 字符串结果
        """
        try:
            if tool_name == "search_jobs":
                return self.search_jobs(
                    keyword=arguments.get("keyword", ""),
                    city=arguments.get("city", "全国"),
                )
            elif tool_name == "re_evaluate_resume":
                return self.re_evaluate_resume(
                    optimized_resume_text=arguments.get("optimized_resume_text", ""),
                )
            elif tool_name == "compare_with_jd":
                return self.compare_with_jd(
                    jd_text=arguments.get("jd_text", ""),
                )
            else:
                return json.dumps(
                    {"error": f"未知工具: {tool_name}"},
                    ensure_ascii=False,
                )
        except Exception as e:
            print(f"[ToolExecutor] 工具 {tool_name} 执行失败: {e}")
            return json.dumps(
                {"error": f"工具执行失败: {str(e)}"},
                ensure_ascii=False,
            )

    # --------------------------------------------------
    # 工具 1: 搜索岗位
    # --------------------------------------------------

    def search_jobs(self, keyword: str, city: str = "全国") -> str:
        """
        使用 duckduckgo_search 搜索招聘信息

        Args:
            keyword: 搜索关键词（岗位名称）
            city: 城市（默认全国）

        Returns:
            JSON 字符串，包含 top 5 结果
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return json.dumps(
                {"error": "搜索功能暂不可用（duckduckgo-search 未安装）"},
                ensure_ascii=False,
            )

        query = f"{keyword} {city} 校招 实习 招聘"
        print(f"[ToolExecutor] search_jobs: query={query}")

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, region="cn-zh", max_results=5))

            jobs = []
            for r in results:
                jobs.append({
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })

            return json.dumps(
                {"keyword": keyword, "city": city, "results": jobs},
                ensure_ascii=False,
            )
        except Exception as e:
            print(f"[ToolExecutor] 搜索失败: {e}")
            return json.dumps(
                {"error": f"搜索失败: {str(e)}", "keyword": keyword, "city": city},
                ensure_ascii=False,
            )

    # --------------------------------------------------
    # 工具 2: 重新评估简历
    # --------------------------------------------------

    def re_evaluate_resume(self, optimized_resume_text: str) -> str:
        """
        调用 IncrementalConvergence + HayCalculator + AbilityMapper 重新评估

        Args:
            optimized_resume_text: 优化后的简历文本

        Returns:
            JSON 字符串，包含新评估结果及与原始评估的对比
        """
        if not self.convergence_engine:
            return json.dumps(
                {"error": "评估引擎不可用"},
                ensure_ascii=False,
            )

        # 从 conversation_memory 获取岗位信息
        memory = self.conversation_memory
        job_title = memory.user_preferences.get("job_title", "")
        job_function = memory.user_preferences.get("job_function", "")

        # 从原始评测中获取（优先）
        if self._original_assessment:
            job_title = job_title or self._original_assessment.get("jobTitle", "未知")
            job_function = job_function or self._original_assessment.get("jobFunction", "未知")

        if not job_title or not job_function:
            return json.dumps(
                {"error": "缺少岗位信息（jobTitle/jobFunction），无法重新评估"},
                ensure_ascii=False,
            )

        print(f"[ToolExecutor] re_evaluate_resume: title={job_title}, function={job_function}")

        try:
            # 调用收敛引擎
            convergence_result = self.convergence_engine.find_optimal_solution(
                eval_text=optimized_resume_text,
                title=job_title,
                function=job_function,
                revenue_contribution={"type": "not_quantifiable"},
                assessment_type="CV",
            )

            if not convergence_result or not convergence_result.get("best_solution"):
                return json.dumps(
                    {"error": "收敛引擎未能生成有效方案，请确保简历内容充分"},
                    ensure_ascii=False,
                )

            best = convergence_result["best_solution"]

            # 提取 HAY 8 因素
            factors = {
                "practical_knowledge": best.get("practical_knowledge", "D"),
                "managerial_knowledge": best.get("managerial_knowledge", "I"),
                "communication": best.get("communication", "2"),
                "thinking_environment": best.get("thinking_environment", "D"),
                "thinking_challenge": best.get("thinking_challenge", "3"),
                "freedom_to_act": best.get("freedom_to_act", "C"),
                "magnitude": best.get("magnitude", "N"),
                "nature_of_impact": best.get("nature_of_impact", "III"),
            }

            # 计算 HAY 评分和职级
            from calculator import calculate_hay_evaluation
            hay_result = calculate_hay_evaluation(factors)
            job_grade = hay_result["summary"].get("job_grade", 14)
            total_score = hay_result["summary"].get("total_score", 0)

            # 学生版：职级下限兜底到 9
            if job_grade < 9:
                job_grade = 9

            # 5 能力映射
            from ability_mapper import map_hay_to_5_abilities
            abilities = map_hay_to_5_abilities(factors)

            # 构建结果
            new_result = {
                "grade": job_grade,
                "total_score": total_score,
                "factors": factors,
                "abilities": {
                    name: {"score": info.get("score"), "level": info.get("level")}
                    for name, info in abilities.items()
                },
            }

            # 与原始评测对比
            comparison = None
            if self._original_assessment:
                orig_grade = self._original_assessment.get("grade")
                orig_abilities = self._original_assessment.get("abilities", {})

                grade_change = None
                if orig_grade is not None:
                    try:
                        grade_change = job_grade - int(orig_grade)
                    except (ValueError, TypeError):
                        pass

                ability_changes = {}
                for name, new_info in abilities.items():
                    old_info = orig_abilities.get(name, {})
                    old_score = old_info.get("score")
                    new_score = new_info.get("score")
                    if old_score is not None and new_score is not None:
                        ability_changes[name] = {
                            "old_score": old_score,
                            "new_score": new_score,
                            "change": new_score - old_score,
                        }

                comparison = {
                    "original_grade": orig_grade,
                    "new_grade": job_grade,
                    "grade_change": grade_change,
                    "ability_changes": ability_changes,
                }

            result = {
                "evaluation": new_result,
                "comparison_with_original": comparison,
            }

            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            print(f"[ToolExecutor] 重评估失败: {e}")
            return json.dumps(
                {"error": f"重新评估失败: {str(e)}"},
                ensure_ascii=False,
            )

    # --------------------------------------------------
    # 工具 3: 对比 JD
    # --------------------------------------------------

    def compare_with_jd(self, jd_text: str) -> str:
        """
        LLM 分析简历与 JD 匹配度

        Args:
            jd_text: JD 原文

        Returns:
            JSON 字符串，包含匹配度分析结果
        """
        if not self.llm_service:
            return json.dumps(
                {"error": "LLM 服务不可用"},
                ensure_ascii=False,
            )

        # 从原始评测中获取简历文本
        resume_text = ""
        if self._original_assessment:
            resume_text = self._original_assessment.get("resume_text", "")

        if not resume_text:
            return json.dumps(
                {"error": "未找到简历内容，无法对比"},
                ensure_ascii=False,
            )

        resume_preview = resume_text[:3000]
        jd_preview = jd_text[:2000]

        prompt = f"""请对比以下简历和 JD（岗位描述），从以下维度分析匹配度：

1. **整体匹配度**：用百分比和等级（高度匹配/较好匹配/一般匹配/较低匹配）表示
2. **优势匹配**：简历中与 JD 要求高度吻合的 2-3 个点
3. **差距分析**：JD 要求但简历中缺失或薄弱的 2-3 个点
4. **优化建议**：针对差距，给出具体的简历改进建议

输出严格的 JSON 格式：
{{
    "match_percentage": 75,
    "match_level": "较好匹配",
    "strengths": ["优势1", "优势2"],
    "gaps": ["差距1", "差距2"],
    "suggestions": ["建议1", "建议2"]
}}

【简历内容】
{resume_preview}

【JD 内容】
{jd_preview}"""

        print(f"[ToolExecutor] compare_with_jd: JD 长度={len(jd_text)}")

        try:
            response = self.llm_service.client.chat.completions.create(
                model=self.llm_service.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的招聘顾问，擅长分析简历与 JD 的匹配度。请输出纯 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content.strip()
            # 验证 JSON 合法性
            result = json.loads(result_text)
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            print(f"[ToolExecutor] JD 对比失败: {e}")
            return json.dumps(
                {"error": f"JD 对比分析失败: {str(e)}"},
                ensure_ascii=False,
            )
