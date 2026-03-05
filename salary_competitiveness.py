"""
学生版薪酬竞争力百分位计算模块

比较对象：同届校招候选人（非全量职场人士）
基于职能和评估出的职级，计算学生在同届校招群体中的百分位排名。

级别范围：8-15 级（学生群体实际分布区间）
各级别人数比例基于学生简历质量分布手动标定：
  8级(8%)  简历信息很少，无实习
  9级(15%) 有经历但描述笼统
  10级(22%) 有实习但深度和量化不够
  11级(25%) 有实习+项目，表达中等（峰值）
  12级(16%) 经历丰富，表达较好
  13级(8%)  名企实习+量化成果
  14级(4%)  多段高质量实习+独立项目
  15级(2%)  顶尖选手（竞赛/创业/头部核心岗）
"""

# 学生群体各级别人数占比（总和 100%）
STUDENT_DISTRIBUTION = {
    8:   8,
    9:  15,
    10: 22,
    11: 25,
    12: 16,
    13:  8,
    14:  4,
    15:  2,
}

MIN_GRADE = 8
MAX_GRADE = 15


def calculate_salary_competitiveness(job_function: str, job_grade: int) -> int:
    """
    计算薪酬竞争力百分位数（学生版）。

    比较对象为同届校招候选人，非全量职场人士。

    Args:
        job_function: 职能名称（当前版本未按职能差异化，预留参数）
        job_grade: 评估出的职级

    Returns:
        0-100 的百分位数，表示"超过了XX%的同届校招候选人"
    """
    # 限制在学生有效范围内
    job_grade = max(MIN_GRADE, min(MAX_GRADE, job_grade))

    # 累加该级别以下的人数占比
    below_pct = sum(
        STUDENT_DISTRIBUTION[g]
        for g in range(MIN_GRADE, job_grade)
    )

    return max(0, min(100, below_pct))
