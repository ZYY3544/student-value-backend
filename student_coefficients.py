"""
===========================================
学生版薪酬修正系数
===========================================
根据学校层级和学历等级对基础薪酬进行修正

薪酬 = 基础薪酬 × 学校系数 × 学历系数
"""

# 学校层级 → 系数
SCHOOL_TIER_COEFFICIENTS = {
    "985":    1.15,
    "211":    1.08,
    "普通本科": 1.00,
    "大专":   0.85,
}

# 学历等级 → 系数
EDUCATION_LEVEL_COEFFICIENTS = {
    "博士": 1.25,
    "硕士": 1.15,
    "本科": 1.00,
    "大专": 0.85,
}


def get_school_coefficient(school_tier: str) -> float:
    """获取学校层级系数"""
    return SCHOOL_TIER_COEFFICIENTS.get(school_tier, 1.00)


def get_education_coefficient(education_level: str) -> float:
    """获取学历等级系数"""
    return EDUCATION_LEVEL_COEFFICIENTS.get(education_level, 1.00)


def apply_student_coefficients(
    base_low: float,
    base_high: float,
    school_tier: str,
    education_level: str,
) -> tuple:
    """
    对基础薪酬区间施加学校+学历修正

    Args:
        base_low: 基础薪酬下限（万元）
        base_high: 基础薪酬上限（万元）
        school_tier: 学校层级 (985/211/普通本科/大专)
        education_level: 学历等级 (博士/硕士/本科/大专)

    Returns:
        (adjusted_low, adjusted_high) 修正后薪酬（万元）
    """
    sc = get_school_coefficient(school_tier)
    ec = get_education_coefficient(education_level)
    factor = sc * ec
    return (base_low * factor, base_high * factor)


def format_salary_k(low_wan: float, high_wan: float) -> str:
    """
    将万元薪酬转为 k 格式

    e.g. 25.6万 → 256k, 30.1万 → 301k
    薪酬格式: "256k-301k"
    """
    low_k = round(low_wan * 10)
    high_k = round(high_wan * 10)
    return f"{low_k}k-{high_k}k"


if __name__ == "__main__":
    # 测试用例
    tests = [
        (15, 20, "985", "硕士", 1.15 * 1.15),
        (12, 16, "211", "本科", 1.08 * 1.00),
        (10, 14, "普通本科", "大专", 1.00 * 0.85),
        (10, 14, "大专", "大专", 0.85 * 0.85),
    ]
    for low, high, tier, edu, expected_factor in tests:
        adj_low, adj_high = apply_student_coefficients(low, high, tier, edu)
        salary_str = format_salary_k(adj_low, adj_high)
        print(f"  {tier}/{edu}: {low}万-{high}万 × {expected_factor:.4f} → {salary_str}")
