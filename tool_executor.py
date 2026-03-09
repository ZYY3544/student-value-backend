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
        # 简历版本管理
        self._resume_versions = {}
        # 外部回调：保存版本时通知 SessionManager 持久化（由 ChatAgent 注入）
        self._on_version_saved = None

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
            elif tool_name == "tailor_resume_to_jd":
                return self.tailor_resume_to_jd(
                    jd_text=arguments.get("jd_text", ""),
                )
            elif tool_name == "verify_job_posting":
                return self.verify_job_posting(
                    jd_text=arguments.get("jd_text", ""),
                )
            elif tool_name == "compare_multiple_jds":
                return self.compare_multiple_jds(
                    jd_list=arguments.get("jd_list", []),
                )
            elif tool_name == "save_resume_version":
                return self.save_resume_version(
                    label=arguments.get("label", ""),
                    resume_text=arguments.get("resume_text", ""),
                    target_jd=arguments.get("target_jd", ""),
                )
            elif tool_name == "list_resume_versions":
                return self.list_resume_versions()
            elif tool_name == "switch_resume_version":
                return self.switch_resume_version(
                    version_id=arguments.get("version_id", ""),
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
        多源搜索招聘信息：Google Custom Search API → LLM 兜底

        Args:
            keyword: 搜索关键词（岗位名称）
            city: 城市（默认全国）

        Returns:
            JSON 字符串，包含搜索结果
        """
        city_text = city if city != "全国" else ""
        print(f"[ToolExecutor] search_jobs: keyword={keyword}, city={city}")

        # Google Custom Search
        # 注：CSE 站点限制需在 Google CSE 控制台配置，这里关键词优化覆盖面
        query = f"{keyword} {city_text} 校招 招聘"
        results = self._google_search(query, num=8)

        for r in results:
            url = r.get("link", "")
            if any(d in url for d in ("zhipin.com", "liepin.com", "lagou.com",
                                       "51job.com", "zhaopin.com", "shixiseng.com")):
                r["source_type"] = "JD"
            elif "nowcoder.com" in url:
                r["source_type"] = "面经"
            elif any(d in url for d in ("xiaohongshu.com", "zhihu.com", "douban.com")):
                r["source_type"] = "求职经验"
            elif "yingjiesheng.com" in url:
                r["source_type"] = "校招资讯"
            else:
                r["source_type"] = "其他"
            # 可信度评估
            credibility = self._assess_credibility(
                url, r.get("title", ""), r.get("snippet", "")
            )
            r["trust_level"] = credibility["trust_level"]
            if credibility["warnings"]:
                r["warnings"] = credibility["warnings"]

        if results:
            return json.dumps(
                {"keyword": keyword, "city": city, "source": "google_search", "results": results[:8]},
                ensure_ascii=False,
            )

        # Google 无结果 → LLM 兜底
        print(f"[ToolExecutor] 搜索结果不理想，使用 LLM 兜底")
        return self._llm_search_fallback(keyword, city)

    def _google_search(self, query: str, num: int = 8) -> list:
        """调用 Google Custom Search JSON API"""
        api_key = os.environ.get('GOOGLE_API_KEY')
        cx = os.environ.get('GOOGLE_CSE_ID')
        if not api_key or not cx:
            return []
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": api_key, "cx": cx, "q": query, "num": min(num, 10)}
        try:
            resp = requests.get(url, params=params, timeout=10, proxies={"http": None, "https": None})
            resp.raise_for_status()
            data = resp.json()
            return [
                {"title": r["title"], "link": r["link"], "snippet": r.get("snippet", "")}
                for r in data.get("items", [])
            ]
        except Exception as e:
            print(f"[ToolExecutor] Google 搜索失败: {e}")
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

        prompt = f"""请对比以下简历和 JD（岗位描述），做深度匹配分析：

1. **整体匹配度**：用百分比和等级（高度匹配/较好匹配/一般匹配/较低匹配）表示
2. **JD 底层能力要求解析**：从 JD 中提取 3-5 个核心能力要求，每个说明要求等级
3. **优势匹配**：简历中与 JD 要求高度吻合的 2-3 个点
4. **差距分析**：JD 要求但简历中缺失或薄弱的 2-3 个点
5. **针对性改写建议**：针对每个差距，指出简历中哪个具体段落可以强化，并给出改写方向

输出严格的 JSON 格式：
{{
    "match_percentage": 75,
    "match_level": "较好匹配",
    "required_abilities": [
        {{"ability": "数据分析能力", "level": "熟练", "jd_evidence": "JD中原文依据"}},
        {{"ability": "跨部门协作", "level": "有经验", "jd_evidence": "JD中原文依据"}}
    ],
    "strengths": ["优势1", "优势2"],
    "gaps": [
        {{"gap": "缺少XX能力体现", "importance": "高/中/低", "resume_section": "可强化的简历段落名"}},
        {{"gap": "缺少YY经验", "importance": "高/中/低", "resume_section": "可强化的简历段落名"}}
    ],
    "tailored_suggestions": [
        {{"section": "实习经历-XX公司", "current_issue": "当前问题", "rewrite_direction": "改写方向", "target_ability": "对应JD要求的能力"}}
    ]
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
            # 绕过系统代理，直接连接目标网站
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=10) as resp:
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

    # --------------------------------------------------
    # 工具 5: 一站式 JD 定制改简历
    # --------------------------------------------------

    def tailor_resume_to_jd(self, jd_text: str) -> str:
        """
        一站式链路：解析 JD → 比对简历 → 生成定制化改写建议

        不只是分析匹配度，而是直接输出改写后的简历段落。
        """
        if not self.llm_service:
            return json.dumps({"error": "LLM 服务不可用"}, ensure_ascii=False)

        resume_text = ""
        if self._original_assessment:
            resume_text = self._original_assessment.get("resume_text", "")
        if not resume_text:
            return json.dumps({"error": "未找到简历内容"}, ensure_ascii=False)

        resume_preview = resume_text[:3000]
        jd_preview = jd_text[:2000]

        prompt = f"""你是一位简历定制专家。请完成以下三步一站式任务：

**第一步：解析 JD 核心要求**
从 JD 中提取 3-5 个核心能力要求。

**第二步：识别简历差距**
对比简历与 JD 要求，找出 2-3 个最关键的差距。

**第三步：直接改写**
针对每个差距，直接给出简历对应段落的改写版本（不是建议，是完成改写后的文本）。
改写规则：
- 基于简历已有信息润色，不编造经历
- 用 STAR 结构强化
- 突出与 JD 匹配的关键词和能力
- 缺少数据的地方用 [待补充: 具体数字] 标记

输出严格 JSON：
{{
    "jd_summary": "JD 一句话概括",
    "core_requirements": [
        {{"ability": "能力名", "importance": "核心/重要/加分"}}
    ],
    "tailored_rewrites": [
        {{
            "section": "简历段落名",
            "original": "原文（简短引用）",
            "rewritten": "改写后的完整段落文本（可直接替换）",
            "improvement": "这次改写强化了什么能力，与 JD 的哪条要求匹配"
        }}
    ],
    "overall_match_after": "改写后预估匹配度提升（如 60% → 78%）"
}}

【简历内容】
{resume_preview}

【JD 内容】
{jd_preview}"""

        print(f"[ToolExecutor] tailor_resume_to_jd: JD 长度={len(jd_text)}")

        try:
            response = self.llm_service.client.chat.completions.create(
                model=self.llm_service.model,
                messages=[
                    {"role": "system", "content": "你是一位简历定制专家，擅长根据 JD 要求精准改写简历。输出纯 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content.strip())
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            print(f"[ToolExecutor] 一站式定制失败: {e}")
            return json.dumps({"error": f"定制改写失败: {str(e)}"}, ensure_ascii=False)

    # --------------------------------------------------
    # 工具 6: 岗位真伪识别
    # --------------------------------------------------

    # 可信域名评级
    _DOMAIN_TRUST = {
        # 高可信度：官方招聘平台
        "zhipin.com": ("高", "Boss直聘官方平台"),
        "liepin.com": ("高", "猎聘官方平台"),
        "lagou.com": ("高", "拉勾官方平台"),
        "51job.com": ("高", "前程无忧官方平台"),
        "zhaopin.com": ("高", "智联招聘官方平台"),
        "shixiseng.com": ("高", "实习僧官方平台"),
        "yingjiesheng.com": ("高", "应届生求职网"),
        "linkedin.com": ("高", "LinkedIn 职业社交平台"),
        "maimai.cn": ("中高", "脉脉职场社交平台"),
        # 中等可信度：社区类内容
        "nowcoder.com": ("中", "牛客社区，面经为主，内容由用户发布"),
        "xiaohongshu.com": ("中", "小红书社区，求职经验分享"),
        "zhihu.com": ("中", "知乎，求职讨论为主"),
        "douban.com": ("中", "豆瓣，求职小组讨论"),
        # 低可信度：需警惕
        "58.com": ("低", "58同城，虚假信息较多，需仔细核实"),
        "ganji.com": ("低", "赶集网，虚假信息较多，需仔细核实"),
    }

    # 虚假招聘常见关键词（扩展版）
    _FRAUD_KEYWORDS = [
        # 培训机构伪装
        "包就业", "零基础", "学费", "培训费", "报名费", "押金",
        "包分配", "先学后付", "免费培训", "转行培训", "IT培训",
        # 常见骗局
        "兼职日结", "刷单", "打字员", "在家办公月入过万",
        "无需经验月薪上万", "转账", "保证金", "入职体检费",
        "日薪500", "日薪1000", "躺赚", "轻松月入",
        # 中介陷阱
        "挂靠", "代缴社保", "人力资源外包招聘",
    ]

    def _assess_credibility(self, url: str, title: str, snippet: str) -> dict:
        """评估单条搜索结果的可信度"""
        trust_level = "中"
        trust_reason = "来源可信度一般"
        warnings = []

        # 域名信任度
        for domain, (level, reason) in self._DOMAIN_TRUST.items():
            if domain in url:
                trust_level = level
                trust_reason = reason
                break

        # 关键词扫描
        combined = (title + " " + snippet).lower()
        for kw in self._FRAUD_KEYWORDS:
            if kw in combined:
                trust_level = "低"
                warnings.append(f"包含可疑关键词「{kw}」")

        # 薪资异常检测（校招月薪超过 5 万大概率有问题）
        import re as _re
        salary_match = _re.search(r'(\d+)[kK万]?\s*[-~至到]\s*(\d+)[kK万]?', combined)
        if salary_match:
            try:
                high = int(salary_match.group(2))
                if 'k' in combined[salary_match.start():salary_match.end()].lower():
                    high *= 1000
                if '万' in combined[salary_match.start():salary_match.end()]:
                    high *= 10000
                if high > 50000:
                    warnings.append("薪资范围异常偏高，请核实")
            except (ValueError, IndexError):
                pass

        if warnings:
            trust_level = "低"

        return {
            "trust_level": trust_level,
            "trust_reason": trust_reason,
            "warnings": warnings,
        }

    def verify_job_posting(self, jd_text: str) -> str:
        """
        使用 LLM 分析一份 JD 是否存在虚假招聘特征
        """
        if not self.llm_service:
            return json.dumps({"error": "LLM 服务不可用"}, ensure_ascii=False)

        jd_preview = jd_text[:2000]

        prompt = f"""请分析以下岗位描述是否存在虚假招聘的特征。

从以下维度检查：
1. 是否要求求职者缴纳费用（培训费、押金、报名费等）
2. 薪资是否与岗位要求明显不匹配（要求低但薪资极高）
3. 公司信息是否模糊（无明确公司名、地址不详）
4. 是否有培训机构伪装招聘的迹象（"包就业"、"零基础转行"）
5. 岗位描述是否过于空泛、缺少具体工作内容
6. 是否存在常见骗局话术

输出 JSON：
{{
    "credibility_score": 85,
    "risk_level": "低风险/中风险/高风险",
    "is_likely_genuine": true,
    "risk_factors": ["风险因素1（如有）"],
    "positive_signals": ["可信信号1"],
    "advice": "一句话建议"
}}

【JD 内容】
{jd_preview}"""

        try:
            response = self.llm_service.client.chat.completions.create(
                model=self.llm_service.model,
                messages=[
                    {"role": "system", "content": "你是一位招聘市场安全专家，擅长识别虚假招聘。输出纯 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content.strip())
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            print(f"[ToolExecutor] 真伪识别失败: {e}")
            return json.dumps({"error": f"真伪识别失败: {str(e)}"}, ensure_ascii=False)

    # --------------------------------------------------
    # 工具 7: 多 JD 横向对比
    # --------------------------------------------------

    def compare_multiple_jds(self, jd_list: list) -> str:
        """
        横向对比多个 JD 与简历的匹配度，推荐最适合的岗位

        Args:
            jd_list: [{"title": "岗位名", "jd_text": "JD全文"}, ...]
        """
        if not self.llm_service:
            return json.dumps({"error": "LLM 服务不可用"}, ensure_ascii=False)

        resume_text = ""
        if self._original_assessment:
            resume_text = self._original_assessment.get("resume_text", "")
        if not resume_text:
            return json.dumps({"error": "未找到简历内容"}, ensure_ascii=False)

        if not jd_list or len(jd_list) < 2:
            return json.dumps({"error": "至少需要 2 个 JD 进行对比"}, ensure_ascii=False)

        resume_preview = resume_text[:2500]
        jds_text = ""
        for i, jd in enumerate(jd_list[:4]):  # 最多 4 个
            title = jd.get("title", f"岗位{i+1}")
            text = jd.get("jd_text", "")[:1200]
            jds_text += f"\n\n--- JD {i+1}: {title} ---\n{text}"

        prompt = f"""请将以下简历与多个 JD 进行横向对比，找出最匹配的岗位。

对每个 JD：
1. 计算匹配度百分比
2. 列出 2 个优势和 2 个差距
3. 给出投递优先级建议

最后给出推荐排序。

输出 JSON：
{{
    "comparisons": [
        {{
            "jd_title": "岗位名",
            "match_percentage": 75,
            "match_level": "较好匹配",
            "strengths": ["优势1", "优势2"],
            "gaps": ["差距1", "差距2"],
            "priority": "推荐投递/可以尝试/匹配度较低"
        }}
    ],
    "recommendation": "综合建议（1-2句话，推荐先投哪个、为什么）",
    "ranking": ["岗位名1（最推荐）", "岗位名2", "岗位名3"]
}}

【简历内容】
{resume_preview}

【JD 列表】
{jds_text}"""

        try:
            response = self.llm_service.client.chat.completions.create(
                model=self.llm_service.model,
                messages=[
                    {"role": "system", "content": "你是一位资深招聘顾问。输出纯 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content.strip())
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            print(f"[ToolExecutor] 多JD对比失败: {e}")
            return json.dumps({"error": f"多JD对比失败: {str(e)}"}, ensure_ascii=False)

    # --------------------------------------------------
    # 工具 8-10: 多版本简历管理
    # --------------------------------------------------

    def save_resume_version(self, label: str, resume_text: str, target_jd: str = "") -> str:
        """保存当前简历为一个命名版本"""
        import time as _time

        version_id = f"v_{len(self._resume_versions) + 1}"
        self._resume_versions[version_id] = {
            "label": label,
            "resume_text": resume_text,
            "target_jd": target_jd[:500] if target_jd else "",
            "created_at": _time.time(),
        }

        # 通知 SessionManager 持久化到 Supabase
        if self._on_version_saved:
            try:
                self._on_version_saved(version_id, label, resume_text, target_jd)
            except Exception as e:
                print(f"[ToolExecutor] 版本持久化回调失败: {e}")

        return json.dumps({
            "version_id": version_id,
            "label": label,
            "message": f"简历版本「{label}」已保存，版本号: {version_id}",
            "total_versions": len(self._resume_versions),
        }, ensure_ascii=False)

    def list_resume_versions(self) -> str:
        """列出所有已保存的简历版本"""
        if not hasattr(self, '_resume_versions') or not self._resume_versions:
            return json.dumps({
                "versions": [],
                "message": "还没有保存任何简历版本。优化完简历后可以保存为不同投递方向的版本。"
            }, ensure_ascii=False)

        versions = []
        for vid, info in self._resume_versions.items():
            versions.append({
                "version_id": vid,
                "label": info["label"],
                "target_jd": info["target_jd"][:100] if info.get("target_jd") else "",
                "resume_preview": info["resume_text"][:200] + "...",
            })

        return json.dumps({
            "versions": versions,
            "total": len(versions),
        }, ensure_ascii=False)

    def switch_resume_version(self, version_id: str) -> str:
        """切换到指定版本的简历"""
        if not hasattr(self, '_resume_versions') or version_id not in self._resume_versions:
            return json.dumps({"error": f"版本 {version_id} 不存在"}, ensure_ascii=False)

        version = self._resume_versions[version_id]
        # 通过 conversation_memory 通知 Agent 切换了版本
        if self.conversation_memory:
            self.conversation_memory.set_user_preference(
                "current_version", f"{version['label']} ({version_id})"
            )

        return json.dumps({
            "version_id": version_id,
            "label": version["label"],
            "resume_text": version["resume_text"],
            "message": f"已切换到简历版本「{version['label']}」",
        }, ensure_ascii=False)
