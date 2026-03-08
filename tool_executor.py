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
import os
import re
import urllib.request
import urllib.error
from html.parser import HTMLParser
from typing import Optional

import requests


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
            elif tool_name == "fetch_url":
                return self.fetch_url(
                    url=arguments.get("url", ""),
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
        多源搜索招聘信息：Bing API → DuckDuckGo → LLM 兜底

        Args:
            keyword: 搜索关键词（岗位名称）
            city: 城市（默认全国）

        Returns:
            JSON 字符串，包含搜索结果
        """
        city_text = city if city != "全国" else ""
        print(f"[ToolExecutor] search_jobs: keyword={keyword}, city={city}")

        # 优先用 Bing 多源搜索
        queries = [
            f"{keyword} {city_text} 校招 JD site:zhipin.com OR site:liepin.com",
            f"{keyword} {city_text} 校招 面经 site:nowcoder.com",
            f"{keyword} 校招 经验 site:xiaohongshu.com",
        ]

        all_results = []
        for q in queries:
            results = self._bing_search(q, count=3)
            for r in results:
                url = r.get("link", "")
                if "zhipin.com" in url or "liepin.com" in url:
                    r["source_type"] = "JD"
                elif "nowcoder.com" in url:
                    r["source_type"] = "面经"
                elif "xiaohongshu.com" in url:
                    r["source_type"] = "求职经验"
                else:
                    r["source_type"] = "其他"
            all_results.extend(results)

        # 去重（按 URL）
        seen = set()
        unique = [r for r in all_results if r["link"] not in seen and not seen.add(r["link"])]

        if unique:
            return json.dumps(
                {"keyword": keyword, "city": city, "source": "bing_search", "results": unique[:8]},
                ensure_ascii=False,
            )

        # Bing 无结果 → LLM 兜底
        print(f"[ToolExecutor] 搜索结果不理想，使用 LLM 兜底")
        return self._llm_search_fallback(keyword, city)

    def _bing_search(self, query: str, count: int = 5) -> list:
        """调用 Bing Web Search API"""
        api_key = os.environ.get('BING_SEARCH_API_KEY')
        if not api_key:
            return []
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        params = {"q": query, "count": count, "mkt": "zh-CN", "textFormat": "Raw"}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [
                {"title": r["name"], "link": r["url"], "snippet": r.get("snippet", "")}
                for r in data.get("webPages", {}).get("value", [])
            ]
        except Exception as e:
            print(f"[ToolExecutor] Bing 搜索失败: {e}")
            return []

    def _llm_search_fallback(self, keyword: str, city: str) -> str:
        """当搜索引擎结果不可用时，用 LLM 生成岗位市场信息"""
        if not self.llm_service:
            return json.dumps(
                {"keyword": keyword, "city": city, "source": "unavailable",
                 "results": [], "note": "搜索暂不可用"},
                ensure_ascii=False,
            )

        prompt = f"""请根据你的知识，列出当前{city}{keyword}相关的校招/实习岗位市场信息。

输出严格 JSON 格式：
{{
    "results": [
        {{"company": "公司名", "title": "岗位名", "requirements": "核心要求（1句话）", "salary_range": "薪资范围"}},
        ...
    ],
    "market_insight": "1-2句话概括该岗位在该城市的市场情况"
}}

要求：
- 列出 3-5 个典型岗位，信息要基于真实市场情况，不要编造不存在的公司
- 绝对不要生成任何 URL 链接（不要编造 link、href、url 字段）
- 只输出上面格式中的字段，不要添加额外字段"""

        try:
            response = self.llm_service.client.chat.completions.create(
                model=self.llm_service.model,
                messages=[
                    {"role": "system", "content": "你是一位熟悉中国互联网招聘市场的顾问。输出纯 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content.strip())
            # 移除 LLM 可能编造的链接字段
            for item in result.get("results", []):
                for link_key in ("link", "href", "url"):
                    item.pop(link_key, None)
            result["keyword"] = keyword
            result["city"] = city
            result["source"] = "llm_knowledge"
            result["note"] = "以上信息基于AI知识库整理，非实时搜索结果，建议去招聘网站确认最新情况"
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            print(f"[ToolExecutor] LLM 兜底搜索失败: {e}")
            return json.dumps(
                {"keyword": keyword, "city": city, "source": "error",
                 "error": f"搜索失败: {str(e)}"},
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

    # --------------------------------------------------
    # 工具 4: 抓取网页内容
    # --------------------------------------------------

    def fetch_url(self, url: str) -> str:
        """
        抓取指定 URL 的网页内容，提取正文文本

        用于获取搜索结果中的 JD 全文，让 Agent 做精准分析。
        使用标准库实现，不依赖 requests/bs4。

        Args:
            url: 目标网页 URL

        Returns:
            JSON 字符串，包含提取的正文文本
        """
        if not url:
            return json.dumps({"error": "URL 不能为空"}, ensure_ascii=False)

        print(f"[ToolExecutor] fetch_url: {url}")

        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                # 处理编码
                charset = resp.headers.get_content_charset() or "utf-8"
                html = resp.read().decode(charset, errors="replace")

            # 提取正文
            text = self._html_to_text(html)

            # 截断过长内容（保留前 4000 字符，足够覆盖一份 JD）
            if len(text) > 4000:
                text = text[:4000] + "\n...(内容过长，已截断)"

            if len(text.strip()) < 50:
                return json.dumps(
                    {"url": url, "error": "页面内容过少，可能需要登录或页面加载依赖 JavaScript"},
                    ensure_ascii=False,
                )

            return json.dumps(
                {"url": url, "content": text, "length": len(text)},
                ensure_ascii=False,
            )

        except urllib.error.HTTPError as e:
            print(f"[ToolExecutor] fetch_url HTTP 错误: {e.code}")
            return json.dumps(
                {"url": url, "error": f"HTTP {e.code}，页面无法访问"},
                ensure_ascii=False,
            )
        except urllib.error.URLError as e:
            print(f"[ToolExecutor] fetch_url URL 错误: {e.reason}")
            return json.dumps(
                {"url": url, "error": f"网络错误: {e.reason}"},
                ensure_ascii=False,
            )
        except Exception as e:
            print(f"[ToolExecutor] fetch_url 失败: {e}")
            return json.dumps(
                {"url": url, "error": f"抓取失败: {str(e)}"},
                ensure_ascii=False,
            )

    @staticmethod
    def _html_to_text(html: str) -> str:
        """
        将 HTML 转为纯文本（标准库实现，无需 bs4）

        策略：
        1. 移除 script/style 标签及其内容
        2. 用 HTMLParser 提取文本
        3. 清理多余空白
        """
        # 移除 script 和 style 块
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

        # 用 HTMLParser 提取文本
        class _TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ('script', 'style', 'noscript'):
                    self._skip = True
                elif tag in ('br', 'p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'tr'):
                    self.parts.append('\n')

            def handle_endtag(self, tag):
                if tag in ('script', 'style', 'noscript'):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    self.parts.append(data)

        extractor = _TextExtractor()
        extractor.feed(html)
        text = ''.join(extractor.parts)

        # 清理多余空白：多个连续空行合并为一个
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
