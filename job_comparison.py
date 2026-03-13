"""
===========================================
多岗位对比模块 (Job Comparison)
===========================================
基于同一份 HAY 评估结果，计算面向不同目标岗位的薪酬、匹配度和竞争力。

核心逻辑：
- HAY 评估只跑 1 次（能力底盘不随目标岗位变化）
- 薪酬按岗位查表（salary_calculator）
- 匹配度基于能力结构 × 岗位典型需求
- 竞争力基于职级在同届中的排名
"""

from typing import Dict, List, Optional
from logger import get_module_logger

logger = get_module_logger(__name__)


# ===========================================
# 岗位能力需求画像
# ===========================================
# 每个岗位对5个能力维度的权重需求（权重越高表示该岗位越看重该能力）
# 专业力, 管理力, 合作力, 思辨力, 创新力
JOB_ABILITY_WEIGHTS = {
    "算法":           {"专业力": 0.35, "管理力": 0.05, "合作力": 0.10, "思辨力": 0.25, "创新力": 0.25},
    "软件开发":       {"专业力": 0.30, "管理力": 0.10, "合作力": 0.15, "思辨力": 0.25, "创新力": 0.20},
    "产品管理":       {"专业力": 0.15, "管理力": 0.25, "合作力": 0.25, "思辨力": 0.20, "创新力": 0.15},
    "数据分析与商业智能": {"专业力": 0.25, "管理力": 0.10, "合作力": 0.15, "思辨力": 0.30, "创新力": 0.20},
    "硬件开发":       {"专业力": 0.35, "管理力": 0.10, "合作力": 0.10, "思辨力": 0.25, "创新力": 0.20},
    "信息安全":       {"专业力": 0.35, "管理力": 0.05, "合作力": 0.10, "思辨力": 0.30, "创新力": 0.20},
    "投融资管理":     {"专业力": 0.25, "管理力": 0.20, "合作力": 0.20, "思辨力": 0.25, "创新力": 0.10},
    "战略管理":       {"专业力": 0.15, "管理力": 0.30, "合作力": 0.20, "思辨力": 0.25, "创新力": 0.10},
    "法务":           {"专业力": 0.35, "管理力": 0.10, "合作力": 0.20, "思辨力": 0.25, "创新力": 0.10},
    "人力资源":       {"专业力": 0.15, "管理力": 0.25, "合作力": 0.30, "思辨力": 0.15, "创新力": 0.15},
    "资产管理":       {"专业力": 0.30, "管理力": 0.20, "合作力": 0.15, "思辨力": 0.25, "创新力": 0.10},
    "市场营销":       {"专业力": 0.15, "管理力": 0.15, "合作力": 0.25, "思辨力": 0.15, "创新力": 0.30},
    "销售":           {"专业力": 0.10, "管理力": 0.15, "合作力": 0.35, "思辨力": 0.15, "创新力": 0.25},
    "硬件测试":       {"专业力": 0.30, "管理力": 0.10, "合作力": 0.15, "思辨力": 0.30, "创新力": 0.15},
    "税务":           {"专业力": 0.35, "管理力": 0.10, "合作力": 0.15, "思辨力": 0.25, "创新力": 0.15},
    "内审":           {"专业力": 0.30, "管理力": 0.15, "合作力": 0.20, "思辨力": 0.25, "创新力": 0.10},
    "软件测试":       {"专业力": 0.30, "管理力": 0.10, "合作力": 0.15, "思辨力": 0.30, "创新力": 0.15},
    "产品运营":       {"专业力": 0.15, "管理力": 0.15, "合作力": 0.25, "思辨力": 0.20, "创新力": 0.25},
    "公共关系":       {"专业力": 0.15, "管理力": 0.15, "合作力": 0.35, "思辨力": 0.15, "创新力": 0.20},
    "游戏设计":       {"专业力": 0.20, "管理力": 0.10, "合作力": 0.15, "思辨力": 0.20, "创新力": 0.35},
    "项目管理":       {"专业力": 0.15, "管理力": 0.30, "合作力": 0.25, "思辨力": 0.20, "创新力": 0.10},
    "电商运营":       {"专业力": 0.15, "管理力": 0.15, "合作力": 0.20, "思辨力": 0.20, "创新力": 0.30},
    "风险管理":       {"专业力": 0.30, "管理力": 0.15, "合作力": 0.15, "思辨力": 0.30, "创新力": 0.10},
    "财务管理":       {"专业力": 0.30, "管理力": 0.20, "合作力": 0.15, "思辨力": 0.25, "创新力": 0.10},
    "会计":           {"专业力": 0.35, "管理力": 0.10, "合作力": 0.15, "思辨力": 0.25, "创新力": 0.15},
    "网络教育":       {"专业力": 0.20, "管理力": 0.15, "合作力": 0.30, "思辨力": 0.15, "创新力": 0.20},
    "供应链管理":     {"专业力": 0.20, "管理力": 0.25, "合作力": 0.20, "思辨力": 0.20, "创新力": 0.15},
    "广告":           {"专业力": 0.15, "管理力": 0.10, "合作力": 0.25, "思辨力": 0.15, "创新力": 0.35},
    "采购":           {"专业力": 0.20, "管理力": 0.20, "合作力": 0.25, "思辨力": 0.20, "创新力": 0.15},
    "客户服务":       {"专业力": 0.15, "管理力": 0.10, "合作力": 0.35, "思辨力": 0.20, "创新力": 0.20},
    "物流":           {"专业力": 0.20, "管理力": 0.25, "合作力": 0.20, "思辨力": 0.20, "创新力": 0.15},
    "行政管理":       {"专业力": 0.15, "管理力": 0.20, "合作力": 0.30, "思辨力": 0.15, "创新力": 0.20},
    "IT服务":         {"专业力": 0.30, "管理力": 0.15, "合作力": 0.20, "思辨力": 0.20, "创新力": 0.15},
    "销售运营":       {"专业力": 0.15, "管理力": 0.20, "合作力": 0.25, "思辨力": 0.20, "创新力": 0.20},
    "媒体推广运营":   {"专业力": 0.15, "管理力": 0.10, "合作力": 0.25, "思辨力": 0.15, "创新力": 0.35},
    "通用职能":       {"专业力": 0.20, "管理力": 0.20, "合作力": 0.20, "思辨力": 0.20, "创新力": 0.20},
}

# 默认权重（未知岗位）
DEFAULT_WEIGHTS = {"专业力": 0.20, "管理力": 0.20, "合作力": 0.20, "思辨力": 0.20, "创新力": 0.20}


def calculate_job_match_score(
    abilities: Dict[str, Dict],
    job_function: str,
) -> int:
    """
    计算能力结构与目标岗位的匹配度 (0-100)

    逻辑：
    1. 获取目标岗位对各能力维度的权重需求
    2. 用户的各维度能力分数 × 岗位权重 → 加权分
    3. 与"理想匹配"做归一化 → 匹配度百分制

    Args:
        abilities: 5维能力评估结果 {"专业力": {"score": 70, ...}, ...}
        job_function: 目标岗位职能

    Returns:
        0-100 的匹配度分数
    """
    weights = JOB_ABILITY_WEIGHTS.get(job_function, DEFAULT_WEIGHTS)

    # 计算加权能力分
    weighted_score = 0
    for ability_name, weight in weights.items():
        ability_info = abilities.get(ability_name, {})
        score = ability_info.get("score", 50) if isinstance(ability_info, dict) else 50
        weighted_score += score * weight

    # 归一化到 0-100（最大可能分是 100 * 1.0 = 100）
    match_score = int(weighted_score)
    return max(10, min(100, match_score))


def _get_top_abilities(abilities: Dict[str, Dict], n: int = 2) -> List[str]:
    """获取得分最高的 n 个能力维度名称"""
    sorted_abilities = sorted(
        abilities.items(),
        key=lambda x: x[1].get("score", 0) if isinstance(x[1], dict) else 0,
        reverse=True,
    )
    return [name for name, _ in sorted_abilities[:n]]


def _get_weak_abilities_for_job(abilities: Dict[str, Dict], job_function: str) -> List[str]:
    """获取对于目标岗位来说最需要提升的能力"""
    weights = JOB_ABILITY_WEIGHTS.get(job_function, DEFAULT_WEIGHTS)

    # 计算每个维度的"差距"：岗位权重高但分数低 = 高差距
    gaps = []
    for ability_name, weight in weights.items():
        ability_info = abilities.get(ability_name, {})
        score = ability_info.get("score", 50) if isinstance(ability_info, dict) else 50
        # 差距 = 岗位需求权重 × (100 - 当前分数)
        gap = weight * (100 - score)
        gaps.append((ability_name, gap))

    gaps.sort(key=lambda x: x[1], reverse=True)
    # 返回差距最大的2个（且权重至少 0.15 的）
    return [name for name, gap in gaps[:2] if gap > 5]


def compare_jobs(
    abilities: Dict[str, Dict],
    job_functions: List[str],
    job_grade: int,
    industry: str,
    city: str,
    school_tier: str,
    education_level: str,
    salary_calculator=None,
) -> List[Dict]:
    """
    多岗位对比：基于同一份能力评估，计算各岗位的薪酬、匹配度和竞争力

    Args:
        abilities: 5维能力评估结果
        job_functions: 待对比的岗位职能列表（最多3个）
        job_grade: HAY 职级
        industry: 行业
        city: 城市（已映射为等级，如"一线城市"）
        school_tier: 学校层级
        education_level: 学历
        salary_calculator: SalaryCalculator 实例

    Returns:
        [
            {
                "jobFunction": "产品管理",
                "salaryRange": "25k~35k",
                "matchScore": 85,
                "competitiveness": 70,
                "strengths": ["管理力", "合作力"],
                "gaps": ["专业力"],
            },
            ...
        ]
    """
    from salary_competitiveness import calculate_salary_competitiveness
    from student_coefficients import apply_student_coefficients, format_salary_k

    results = []

    for func in job_functions[:3]:  # 最多3个岗位
        # 1. 匹配度
        match_score = calculate_job_match_score(abilities, func)

        # 2. 薪酬
        salary_range = "暂无数据"
        if salary_calculator:
            try:
                salary_result = salary_calculator.get_salary_range(
                    job_grade=job_grade,
                    function=func,
                    industry=industry,
                    city=city,
                )
                if salary_result:
                    adj_low, adj_high = apply_student_coefficients(
                        salary_result['P50_low'],
                        salary_result['P50_high'],
                        school_tier,
                        education_level,
                    )
                    salary_range = format_salary_k(adj_low, adj_high)
            except Exception as e:
                logger.warning(f"岗位 {func} 薪酬查询失败: {e}")

        # 3. 竞争力
        competitiveness = calculate_salary_competitiveness(func, job_grade)

        # 4. 优势和差距
        top_abilities = _get_top_abilities(abilities)
        weak_abilities = _get_weak_abilities_for_job(abilities, func)

        results.append({
            "jobFunction": func,
            "salaryRange": salary_range,
            "matchScore": match_score,
            "competitiveness": competitiveness,
            "strengths": top_abilities,
            "gaps": weak_abilities,
        })

    # 按匹配度排序（最匹配的排前面）
    results.sort(key=lambda x: x["matchScore"], reverse=True)

    logger.info(f"[JobComparison] 多岗位对比完成: "
                + ", ".join(f"{r['jobFunction']}(匹配{r['matchScore']})" for r in results))

    return results


def get_recommended_job(
    abilities: Dict[str, Dict],
    current_function: str,
) -> Optional[str]:
    """
    根据能力结构推荐最匹配的岗位（从全部岗位中选出匹配度最高且不同于当前岗位的）

    Returns:
        推荐岗位名称，或 None
    """
    best_func = None
    best_score = 0

    for func in JOB_ABILITY_WEIGHTS:
        if func == current_function or func == "通用职能":
            continue
        score = calculate_job_match_score(abilities, func)
        if score > best_score:
            best_score = score
            best_func = func

    return best_func


if __name__ == "__main__":
    # 测试
    test_abilities = {
        "专业力": {"score": 70, "level": "high"},
        "管理力": {"score": 55, "level": "medium"},
        "合作力": {"score": 65, "level": "medium"},
        "思辨力": {"score": 60, "level": "medium"},
        "创新力": {"score": 50, "level": "medium"},
    }

    for func in ["产品管理", "软件开发", "人力资源", "数据分析与商业智能"]:
        score = calculate_job_match_score(test_abilities, func)
        print(f"  {func}: 匹配度 {score}")

    recommended = get_recommended_job(test_abilities, "产品管理")
    print(f"\n  推荐岗位: {recommended}")
