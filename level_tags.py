"""
===========================================
学生版职级标签映射规则（专业版）
===========================================
覆盖 9-15 级，使用专业职场段位 + 分数制呈现。

段位体系（5个段位）：
- 90+   战略引领者
- 80-89 业务驱动者
- 70-79 独立执行者
- 60-69 成长潜力者
- <60   起步探索者

能力评分：基于 HAY 总分映射到 0-100 分制
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

# 职级 → 能力评分区间（0-100 分制）
# 9级对应 30-44, 10级对应 45-54, ..., 15级对应 90-100
GRADE_TO_SCORE_RANGE = {
    9:  (30, 44),
    10: (45, 54),
    11: (55, 64),
    12: (65, 74),
    13: (75, 84),
    14: (85, 92),
    15: (93, 100),
}

# 段位定义：(分数下限, 段位名称, 段位描述)
PROFESSIONAL_TIERS = [
    (90, "战略引领者", "你的经历深度和广度已经展现出战略级视野，具备引领业务方向的潜力。在校招市场中属于顶尖稀缺人才，值得最好的平台。"),
    (75, "业务驱动者", "你具备独立驱动业务的综合实力，经历丰富且有深度。在校招竞争中处于领先梯队，大厂核心岗位是你的舞台。"),
    (60, "独立执行者", "你已经能够独立承担有一定复杂度的工作，具备扎实的专业基础和实战经验。在校招中具有明确的竞争力。"),
    (45, "成长潜力者", "你已经积累了一定的实践经验，具备持续成长的基础。通过针对性地补强经历短板，竞争力还有很大提升空间。"),
    (0,  "起步探索者", "你正处于职业探索的起步阶段，经历积累还在早期。建议通过实习、项目等方式快速丰富实战经验。"),
]


def _hay_total_to_ability_score(job_grade: int, total_score: int) -> int:
    """
    将 HAY 总分映射到 0-100 的能力评分

    映射逻辑：
    1. 根据 job_grade 确定该级的评分区间
    2. 根据 total_score 在该级 HAY 总分区间中的位置，线性插值到评分区间
    """
    if job_grade < 9:
        job_grade = 9
    elif job_grade > 15:
        job_grade = 15

    # 获取 HAY 总分区间
    hay_low, hay_high = GRADE_SCORE_RANGES.get(job_grade, (161, 191))
    # 获取评分区间
    score_low, score_high = GRADE_TO_SCORE_RANGE.get(job_grade, (55, 64))

    # 线性插值
    hay_span = hay_high - hay_low
    if hay_span <= 0:
        return score_low

    ratio = (total_score - hay_low) / hay_span
    ratio = max(0.0, min(1.0, ratio))  # 限制在 [0, 1]

    ability_score = int(score_low + ratio * (score_high - score_low))
    return max(0, min(100, ability_score))


def _get_professional_tier(ability_score: int) -> Tuple[str, str]:
    """根据能力评分返回 (段位名称, 段位描述)"""
    for threshold, name, desc in PROFESSIONAL_TIERS:
        if ability_score >= threshold:
            return name, desc
    return PROFESSIONAL_TIERS[-1][1], PROFESSIONAL_TIERS[-1][2]


def get_student_level_tag_and_desc(
    job_grade: int,
    total_score: int = 0
) -> Tuple[str, str]:
    """
    学生版标签生成函数（专业版）

    Args:
        job_grade: 职级 (9-15)
        total_score: HAY 总分，用于精确计算能力评分

    Returns:
        (tier_name, tier_description) 段位名称和描述
    """
    if job_grade < 9:
        job_grade = 9
    elif job_grade > 15:
        job_grade = 15

    ability_score = _hay_total_to_ability_score(job_grade, total_score)
    return _get_professional_tier(ability_score)


def get_ability_score(job_grade: int, total_score: int = 0) -> int:
    """
    获取能力评分（0-100）

    Args:
        job_grade: 职级 (9-15)
        total_score: HAY 总分

    Returns:
        0-100 的能力评分
    """
    return _hay_total_to_ability_score(job_grade, total_score)


# 保留原版函数签名的兼容包装
def get_level_tag_and_desc(
    job_grade: int,
    factors: Dict[str, str] = None,
    abilities: Dict[str, Dict] = None,
    total_score: int = 0
) -> Tuple[str, str]:
    """
    兼容原版调用签名，内部转发到学生版逻辑
    """
    return get_student_level_tag_and_desc(job_grade, total_score)


if __name__ == "__main__":
    print("=" * 60)
    print("学生版职级标签测试（专业版）")
    print("=" * 60)

    for grade in range(9, 16):
        low, high = GRADE_SCORE_RANGES[grade]
        for sub, label in [("低档", low + 2), ("中档", (low + high) // 2), ("高档", high - 2)]:
            ability_score = _hay_total_to_ability_score(grade, label)
            tier_name, tier_desc = _get_professional_tier(ability_score)
            print(f"\n【{grade}级 {sub}】能力评分: {ability_score} | 段位: {tier_name}")
            print(f"  {tier_desc[:50]}...")
