"""
学生版薪酬竞争力百分位计算模块

基于职能和评估出的职级，计算学生在同届校招群体中的百分位排名。
峰值整体下移至 10-11 级（学生群体典型分布）。
"""

# 学生版：各职能的峰值职级（下移 3-4 级）
DEFAULT_GRADES = {
    "算法": 12, "软件开发": 11, "产品管理": 11, "数据分析与商业智能": 11,
    "硬件开发": 11, "信息安全": 10, "投融资管理": 11, "战略管理": 11,
    "法务": 10, "人力资源": 10, "资产管理": 10, "市场营销": 10,
    "销售": 10, "硬件测试": 10, "税务": 10, "内审": 10,
    "软件测试": 10, "产品运营": 10, "公共关系": 9, "游戏设计": 9,
    "项目管理": 10, "电商运营": 9, "风险管理": 9, "财务管理": 10,
    "会计": 9, "网络教育": 9, "供应链管理": 9, "广告": 9,
    "采购": 9, "客户服务": 9, "物流": 9, "行政管理": 9,
    "IT服务": 9, "销售运营": 9, "媒体推广运营": 9, "通用职能": 10,
}

# 学生版有效级别范围（聚焦 6-18）
MIN_GRADE = 6
MAX_GRADE = 18

# 分布参数
ALPHA = 1.5    # 左侧上升曲率
DECAY = 0.55   # 右侧衰减速率
TOTAL_PEOPLE = 1000  # 归一化基数


def calculate_salary_competitiveness(job_function: str, job_grade: int) -> int:
    """
    计算薪酬竞争力百分位数（学生版）。

    Args:
        job_function: 职能名称
        job_grade: 评估出的职级

    Returns:
        0-100 的百分位数，表示"超过了XX%的同届校招候选人"
    """
    peak = DEFAULT_GRADES.get(job_function, 10)

    # 限制 job_grade 在有效范围内
    job_grade = max(MIN_GRADE, min(MAX_GRADE, job_grade))

    # 构造每个级别的权重（偏态分布）
    weights = {}
    for g in range(MIN_GRADE, MAX_GRADE + 1):
        if g <= peak:
            weights[g] = (g - MIN_GRADE + 1) ** ALPHA
        else:
            peak_weight = (peak - MIN_GRADE + 1) ** ALPHA
            weights[g] = peak_weight * (DECAY ** (g - peak))

    # 归一化
    total_weight = sum(weights.values())
    people = {g: (w / total_weight) * TOTAL_PEOPLE for g, w in weights.items()}

    # 累加用户级别以下的人数
    below_count = sum(people[g] for g in range(MIN_GRADE, job_grade))

    percentile = int(round(below_count / TOTAL_PEOPLE * 100))
    return max(0, min(100, percentile))
