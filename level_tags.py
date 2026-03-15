"""
===========================================
校招简历评估 —— 职业能力层级定义 V3
===========================================
适用范围：大学生校招场景
报告上只显示称谓 + 一段合并描述，不显示级别数字和 band 标签
级别数字仅用于后台计算（薪酬映射等），不暴露给用户
"""

from typing import Dict, Tuple


# 每个职级的 HAY 总分范围（来自 data_tables.py 的 JOB_GRADE_RANGES）
GRADE_SCORE_RANGES = {
    9:  (114, 134),
    10: (135, 160),
    11: (161, 191),
    12: (192, 227),
    13: (228, 268),
    14: (269, 313),
    15: (314, 370),
}

# 职级 → 能力评分区间（0-100 分制，仅后台使用）
GRADE_TO_SCORE_RANGE = {
    9:  (30, 44),
    10: (45, 54),
    11: (55, 64),
    12: (65, 74),
    13: (75, 84),
    14: (85, 92),
    15: (93, 100),
}

# V5 层级定义：一句话定义，个性化解读由 Sparky 在对话中完成
LEVEL_DEFINITIONS = {
    9: {
        "title": "学习准备者",
        "description": "处于知识积累阶段，尚未形成可迁移到工作场景中的实践能力。",
    },
    10: {
        "title": "初始实践者",
        "description": "能够在明确指导下完成基础的辅助性任务，对工作流程有初步认知。",
    },
    11: {
        "title": "基础工作执行者",
        "description": "能够在指导下开展有一定专业门槛的基础工作，对个人工作成果的质量和时效负责。",
    },
    12: {
        "title": "例行工作执行者",
        "description": "能够按照既定流程独立完成例行性的专业工作，不需要逐步指导。",
    },
    13: {
        "title": "探索型工作执行者",
        "description": "在完成基本工作的过程中，能够主动尝试新的方法和思路来提升效率。",
    },
    14: {
        "title": "初步独立贡献者",
        "description": "能够独立承担单一模块的工作职责，对所负责部分的质量和进度有完整把控。",
    },
    15: {
        "title": "独立贡献者",
        "description": "能够独立承担完整业务模块的职责，并指导他人开展工作。",
    },
}


def _hay_total_to_ability_score(job_grade: int, total_score: int) -> int:
    """将 HAY 总分映射到 0-100 的能力评分（仅后台使用）"""
    if job_grade < 9:
        job_grade = 9
    elif job_grade > 15:
        job_grade = 15

    hay_low, hay_high = GRADE_SCORE_RANGES.get(job_grade, (161, 191))
    score_low, score_high = GRADE_TO_SCORE_RANGE.get(job_grade, (55, 64))

    hay_span = hay_high - hay_low
    if hay_span <= 0:
        return score_low

    ratio = (total_score - hay_low) / hay_span
    ratio = max(0.0, min(1.0, ratio))

    ability_score = int(score_low + ratio * (score_high - score_low))
    return max(0, min(100, ability_score))


def get_student_level_tag_and_desc(
    job_grade: int,
    total_score: int = 0
) -> Tuple[str, str]:
    """
    V3 标签生成：按 job_grade 直接映射 title + description

    Returns:
        (title, description) — 报告上显示的称谓和描述
    """
    if job_grade < 9:
        job_grade = 9
    elif job_grade > 15:
        job_grade = 15

    level_def = LEVEL_DEFINITIONS.get(job_grade, LEVEL_DEFINITIONS[11])
    return level_def["title"], level_def["description"]


def get_ability_score(job_grade: int, total_score: int = 0) -> int:
    """获取能力评分 0-100（仅后台使用，不暴露给用户）"""
    return _hay_total_to_ability_score(job_grade, total_score)


# 兼容原版调用签名
def get_level_tag_and_desc(
    job_grade: int,
    factors: Dict[str, str] = None,
    abilities: Dict[str, Dict] = None,
    total_score: int = 0
) -> Tuple[str, str]:
    """兼容原版调用签名，内部转发到 V3 逻辑"""
    return get_student_level_tag_and_desc(job_grade, total_score)
