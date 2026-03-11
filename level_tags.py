"""
===========================================
学生版趣味标签映射规则
===========================================
覆盖 9-15 级，每级拆 3 个子档（-/N/+），共21条标签。
子档依据 HAY 总分在该级分数区间内的位置判定。
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

# (grade, sub_tier) → (tag, description)
# sub_tier: "-" = 低档, "N" = 中档, "+" = 高档
STUDENT_LEVEL_TAGS = {
    (9,  "-"): ("萌新探路者",    "刚踏出校门第一步，世界很大，你的好奇心更大。起点不决定终点，你的故事才刚刚开始写呢。"),
    (9,  "N"): ("校园新秀",      "在同届里已经有了自己的节奏，虽然经验值还在攒，但方向感已经有了。稳住，慢慢来。"),
    (9,  "+"): ("潜力萌芽",      "别看现在低调，你的底子比你想象的扎实。再多实习一两段，简历直接起飞。"),

    (10, "-"): ("入门小能手",    "基本功已经有了，接下来就是找到属于你的主线任务。校招战场上，你不算新手了。"),
    (10, "N"): ("初露锋芒",      "同届里你已经跑在前面了，面试官看你的简历会多停留几秒。保持这个势头，offer会来找你。"),
    (10, "+"): ("实力储备中",    "能力值悄悄在涨，只差一个展示舞台。秋招春招你都有戏，稳稳拿捏。"),

    (11, "-"): ("行业见习生",    "已经摸到行业门道了，比大多数同学多了一份实战嗅觉。多卷一点项目经验，竞争力翻倍不是梦。"),
    (11, "N"): ("赛道领跑者",    "你的综合素质在同届中已经挺突出了，面试的时候自信点，你值得更好的offer。"),
    (11, "+"): ("Offer收割预备",  "简历亮点够多，表达也有章法。再打磨下面试细节，收割offer的日子不远了。"),

    (12, "-"): ("校招硬通货",    "你的简历在HR手里属于优先处理那一档，技术面+综合面都能打。稳住，好offer在路上。"),
    (12, "N"): ("准MVP",          "放在校招赛场上你就是种子选手，专业深度+视野广度都在线。冲刺大厂SP，完全有戏。"),
    (12, "+"): ("校园卷王",      "别人还在准备的时候你已经领先一个身位了。这波属于降维打击，面试官都得正襟危坐。"),

    (13, "-"): ("天选打工人",    "校招top梯队稳稳的，企业抢着给你发offer。你唯一的烦恼是：选哪个。"),
    (13, "N"): ("校招天花板",    "你就是传说中'别人家的应届生'。大厂SSP已经在向你招手了，剩下的交给缘分。"),
    (13, "+"): ("应届传说",      "简直是校招版六边形战士，哪个维度都能打。HR看到你的简历，估计要抢着约面。"),

    (14, "-"): ("学术新锐",      "研究功底扎实，专业深度已经超越大多数同龄人。企业看你的简历，看到的是潜力股中的硬核选手。"),
    (14, "N"): ("科研潜力股",    "论文、项目、实战三线并行，你的知识密度让面试官眼前一亮。大厂研究岗的候选名单上，有你的名字。"),
    (14, "+"): ("硬核研究员",    "专业能力已经逼近资深从业者水平，校招圈里属于降维打击。你不是在找工作，是工作在找你。"),

    (15, "-"): ("顶尖学者胚子",  "博士级别的专业积累加上清晰的产业视野，你在校招市场上是稀缺资源，企业愿意为你开special offer。"),
    (15, "N"): ("产学研全能王",  "学术深度和工程能力双双在线，放眼整个校招池子都是凤毛麟角。你的竞争力，配得上最好的平台。"),
    (15, "+"): ("未来科学家",    "还没毕业就已经站在了行业前沿，你的能力边界已经超出校招的评估范围。未来可期，不，未来已来。"),
}


def _determine_sub_tier(job_grade: int, total_score: int) -> str:
    """
    根据 HAY 总分在该级区间内的位置判定子档

    将分数区间三等分:
    - 下 1/3 → "-"
    - 中 1/3 → "N"
    - 上 1/3 → "+"
    """
    if job_grade not in GRADE_SCORE_RANGES:
        return "N"

    low, high = GRADE_SCORE_RANGES[job_grade]
    span = high - low
    if span <= 0:
        return "N"

    relative = total_score - low
    third = span / 3

    if relative < third:
        return "-"
    elif relative < third * 2:
        return "N"
    else:
        return "+"


def get_student_level_tag_and_desc(
    job_grade: int,
    total_score: int = 0
) -> Tuple[str, str]:
    """
    学生版标签生成函数

    Args:
        job_grade: 职级 (9-15)
        total_score: HAY 总分，用于判定子档

    Returns:
        (tag, description)
    """
    # 下限兜底到 9
    if job_grade < 9:
        job_grade = 9
    # 上限兜底到 15
    elif job_grade > 15:
        job_grade = 15

    sub_tier = _determine_sub_tier(job_grade, total_score)
    key = (job_grade, sub_tier)

    if key in STUDENT_LEVEL_TAGS:
        return STUDENT_LEVEL_TAGS[key]

    # 兜底：使用中档
    return STUDENT_LEVEL_TAGS.get((job_grade, "N"), ("校园新秀", "你的校招之旅正在展开，加油！"))


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
    print("学生版职级标签测试")
    print("=" * 60)

    for grade in range(9, 16):
        low, high = GRADE_SCORE_RANGES[grade]
        for sub, label in [("-", "低档"), ("N", "中档"), ("+", "高档")]:
            # 模拟不同分数
            if sub == "-":
                score = low + 2
            elif sub == "N":
                score = (low + high) // 2
            else:
                score = high - 2
            tag, desc = get_student_level_tag_and_desc(grade, score)
            print(f"\n【{grade}级 {label}】{tag} (score={score})")
            print(f"  {desc}")
