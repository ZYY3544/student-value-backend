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

# V3 层级定义：按 job_grade 直接映射，不再用分数段位
LEVEL_DEFINITIONS = {
    9: {
        "title": "学习准备者",
        "description": (
            "正处于知识积累阶段，还没有形成可迁移的实践经验。"
            "课堂学习和课外活动是当前的主要经历，尚未接触过真实的工作场景或商业环境。"
            "\n\n"
            "大多数大一大二、还没开始实习的同学处于这个阶段。"
            "当务之急不是优化简历措辞，而是先积累一段有实质内容的实习或项目经历。"
        ),
    },
    10: {
        "title": "初始实践者",
        "description": (
            "开始接触真实工作场景，能在明确指导下完成基础的辅助性任务。"
            "对某个领域有初步了解，但还不能独立产出有完整价值的工作成果。"
            "\n\n"
            "做过短期实习、课程项目、或校园组织中有一定执行职责的同学通常在这个阶段。"
            "你已经迈出了第一步，接下来需要争取更深度参与项目核心环节的机会。"
        ),
    },
    11: {
        "title": "基础工作执行者",
        "description": (
            "能在指导下完成有一定专业门槛的基础工作，"
            "对自己负责的任务的完成质量和时效有意识把控。"
            "\n\n"
            "做过一段较完整的实习、承担过明确分工的项目任务的同学通常在这个阶段。"
            "你已经能「把事做完」，下一步是学会「把事做好」——从执行指令到理解指令背后的目的。"
        ),
    },
    12: {
        "title": "例行工作执行者",
        "description": (
            "能按照既定流程独立完成例行性的专业工作，"
            "不需要每一步都有人指导，能对自己的工作成果负责。"
            "\n\n"
            "有过较深度的实习经历、在实习中能独立负责某一块具体工作的同学通常在这个阶段。"
            "你已经能「照着做」，下一步是「自己想办法做得更好」——从遵循流程到优化流程。"
        ),
    },
    13: {
        "title": "探索型工作执行者",
        "description": (
            "在完成基本工作的同时，会主动尝试新的方法和思路来提升效率。"
            "不只是按指令做事，开始有自己的判断和改进意识。"
            "\n\n"
            "在实习中不只是完成交办任务、还主动提出过优化建议或尝试过新工具的同学通常在这个阶段。"
            "你已经展现出从「执行者」向「贡献者」过渡的趋势，但还需要更多独立负责完整项目的经验来验证。"
        ),
    },
    14: {
        "title": "初步独立贡献者",
        "description": (
            "能独立承担一个模块的工作，对自己负责部分的质量和进度有完整的把控。"
            "不只是做好自己的事，开始理解自己的工作在整体中的位置和价值。"
            "\n\n"
            "在实习中独立负责过一个完整模块（比如一份研究报告、一个产品功能、一次用户调研）的同学通常在这个阶段。"
            "校招生能达到这个级别已经很有竞争力了——你有能力在入职后快速独立上手。"
        ),
    },
    15: {
        "title": "独立贡献者",
        "description": (
            "能独立承担一个完整业务模块或大型项目中的核心部分，"
            "并且能指导他人开展工作。对自己负责领域的整体产出负责。"
            "\n\n"
            "极少数校招生能达到这个级别——通常是有过创业经历、带队经历、"
            "或在头部公司长期实习且承担过核心职责的人。"
            "这个级别意味着你入职后可以跳过大部分新人适应期，直接进入产出状态。"
        ),
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
