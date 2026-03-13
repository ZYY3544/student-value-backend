"""
===========================================
Function Call 工具执行器 (Tool Executor)
===========================================
为 OptimizeAgent 提供外部工具调用能力（仅限 LLM 无法直接完成的操作）：
1. search_jobs — 搜索招聘岗位（Tavily API）
2. re_evaluate_resume — 用 HAY 体系重新评估优化后简历（算法管线）
3. fetch_url — 抓取网页内容
4. save/list/switch_resume_version — 简历版本管理
"""

import json
import os
import re
import urllib.request
import urllib.error
from html.parser import HTMLParser

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
            elif tool_name == "fetch_url":
                return self.fetch_url(
                    url=arguments.get("url", ""),
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
        搜索招聘信息（Tavily Search API）

        Args:
            keyword: 搜索关键词（岗位名称）
            city: 城市（默认全国）

        Returns:
            JSON 字符串，包含搜索结果
        """
        city_text = city if city != "全国" else ""
        print(f"[ToolExecutor] search_jobs: keyword={keyword}, city={city}")

        # Tavily Search API（全网搜索）
        query = f"{keyword} {city_text} 校招 招聘"
        results = self._tavily_search(query, max_results=8)

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

        return json.dumps(
            {"keyword": keyword, "city": city, "source": "tavily_search", "results": results[:8]},
            ensure_ascii=False,
        )

    def _tavily_search(self, query: str, max_results: int = 8) -> list:
        """调用 Tavily Search API（全网搜索）"""
        api_key = os.environ.get('TAVILY_API_KEY')
        if not api_key:
            return []
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"query": query, "max_results": min(max_results, 20), "search_depth": "basic"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {"title": r.get("title", ""), "link": r.get("url", ""), "snippet": r.get("content", "")}
                for r in data.get("results", [])
            ]
        except Exception as e:
            print(f"[ToolExecutor] Tavily 搜索失败: {e}")
            return []

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

            # 8 能力维度映射
            from ability_mapper import map_factors_to_dimensions
            abilities = map_factors_to_dimensions(factors)

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
    # 工具 3: 抓取网页内容
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
    # 可信度评估（供 search_jobs 使用）
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

    # --------------------------------------------------
    # 工具 4-6: 多版本简历管理
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
