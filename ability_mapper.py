"""
===========================================
能力映射模块 (Ability Mapper)
===========================================
将HAY 8因素转换为C端用户可理解的5个能力维度

5个能力维度：
1. 专业力 - 技术/专业知识的深度和广度
2. 管理力 - 管理和组织协调能力
3. 合作力 - 沟通协作和影响力
4. 思辨力 - 分析判断和解决问题的能力
5. 创新力 - 创新思维和开拓能力

映射关系：
- 专业力 ← PK(100%)
- 管理力 ← MK(70%) + FTA(30%)
- 合作力 ← Comm(80%) + NI(20%)
- 思辨力 ← TE(100%)
- 创新力 ← TC(100%)
"""

from typing import Dict
from logger import get_module_logger

logger = get_module_logger(__name__)


# ===========================================
# 档位到分数的映射表
# ===========================================

# 专业知识 (PK) 档位映射
PK_SCORE_MAP = {
    'A-': 15, 'A': 20, 'A+': 25,
    'B-': 25, 'B': 30, 'B+': 35,
    'C-': 35, 'C': 40, 'C+': 45,
    'D-': 45, 'D': 50, 'D+': 55,
    'E-': 55, 'E': 60, 'E+': 65,
    'F-': 65, 'F': 70, 'F+': 75,
    'G-': 75, 'G': 80, 'G+': 85,
    'H-': 85, 'H': 90, 'H+': 95,
}

# 管理知识 (MK) 档位映射
MK_SCORE_MAP = {
    'T-': 15, 'T': 20, 'T+': 25,
    'I-': 30, 'I': 35, 'I+': 40,
    'II-': 45, 'II': 50, 'II+': 55,
    'III-': 60, 'III': 65, 'III+': 70,
    'IV-': 75, 'IV': 80, 'IV+': 85,
    'V-': 85, 'V': 90, 'V+': 95,
}

# 沟通技巧 (Comm) 档位映射
COMM_SCORE_MAP = {
    '1': 40,
    '2': 60,
    '3': 80,
}

# 思维环境 (TE) 档位映射 (与PK相同结构)
TE_SCORE_MAP = PK_SCORE_MAP.copy()

# 思维挑战 (TC) 档位映射
TC_SCORE_MAP = {
    '1-': 25, '1': 30, '1+': 35,
    '2-': 40, '2': 45, '2+': 50,
    '3-': 55, '3': 60, '3+': 65,
    '4-': 70, '4': 75, '4+': 80,
    '5-': 85, '5': 90, '5+': 95,
}

# 行动自由 (FTA) 档位映射 (与PK相同结构)
FTA_SCORE_MAP = PK_SCORE_MAP.copy()

# 影响范围 (M) 档位映射
M_SCORE_MAP = {
    'N': 30,  # 不可量化，给基础分
    '1-': 35, '1': 40, '1+': 45,
    '2-': 45, '2': 50, '2+': 55,
    '3-': 55, '3': 60, '3+': 65,
    '4-': 65, '4': 70, '4+': 75,
    '5-': 75, '5': 80, '5+': 85,
    '6-': 85, '6': 90, '6+': 95,
    '7-': 90, '7': 92, '7+': 95,
    '8-': 92, '8': 95, '8+': 97,
    '9-': 95, '9': 97, '9+': 99,
}

# 影响性质 (NI) 档位映射
# I-VI系统（不可量化时使用）
NI_ROMAN_SCORE_MAP = {
    'I-': 30, 'I': 35, 'I+': 40,
    'II-': 40, 'II': 45, 'II+': 50,
    'III-': 50, 'III': 55, 'III+': 60,
    'IV-': 60, 'IV': 65, 'IV+': 70,
    'V-': 70, 'V': 75, 'V+': 80,
    'VI-': 80, 'VI': 85, 'VI+': 90,
}

# RCSP系统（可量化时使用）
NI_RCSP_SCORE_MAP = {
    'R-': 30, 'R': 35, 'R+': 40,
    'C-': 45, 'C': 50, 'C+': 55,
    'S-': 60, 'S': 65, 'S+': 70,
    'P-': 75, 'P': 80, 'P+': 85,
}


# ===========================================
# 能力解释文案
# ===========================================

ABILITY_EXPLANATIONS = {
    "专业力": {
        "high": "专业知识深厚，能解决高复杂度问题",
        "medium": "专业基础扎实，能独立完成常规工作",
        "low": "具备基础专业能力，适合执行标准化流程"
    },
    "管理力": {
        "high": "能统筹多业务领域，具备战略级管理视野",
        "medium": "能管理团队或项目，协调资源达成目标",
        "low": "能管理自身任务，配合团队完成目标"
    },
    "合作力": {
        "high": "能影响高层决策，在复杂环境中建立共识",
        "medium": "能跨部门协作，协调各方利益推动落地",
        "low": "能在团队内有效沟通，配合完成协作任务"
    },
    "思辨力": {
        "high": "能在模糊环境中洞察本质，做出战略判断",
        "medium": "能分析复杂问题，提出有效解决方案",
        "low": "能按既定框架分析解决问题，逻辑清晰"
    },
    "创新力": {
        "high": "能开拓新领域，推动突破性创新",
        "medium": "能改进现有流程和方法，持续优化",
        "low": "能在现有框架下完成工作，学习新事物"
    }
}


# ===========================================
# 核心映射函数
# ===========================================

def _get_level_score(value: str, score_map: Dict[str, int], default: int = 50) -> int:
    """从档位获取分数"""
    if not value:
        return default

    if value in score_map:
        return score_map[value]

    base_value = value.rstrip('+-')
    if base_value in score_map:
        return score_map[base_value]

    logger.warning(f"未找到档位 '{value}' 的映射，使用默认值 {default}")
    return default


def _get_ni_score(ni_value: str, magnitude: str) -> int:
    """获取影响性质(NI)的分数，根据magnitude决定使用RCSP还是罗马数字系统"""
    if not ni_value:
        return 50
    if magnitude == 'N' or magnitude.startswith('N'):
        return _get_level_score(ni_value, NI_ROMAN_SCORE_MAP, 50)
    else:
        return _get_level_score(ni_value, NI_RCSP_SCORE_MAP, 50)


def _get_level_tag(score: int) -> str:
    """根据分数获取等级标签"""
    if score >= 70:
        return 'high'
    elif score >= 45:
        return 'medium'
    else:
        return 'low'


def _weighted_average(items: list) -> int:
    """加权平均"""
    total = sum(score * weight for score, weight in items)
    return round(total)


def map_hay_to_5_abilities(hay_factors: Dict[str, str]) -> Dict[str, Dict]:
    """
    将 HAY 8因素转换为 5 个能力维度（加权合并）

    Args:
        hay_factors: HAY 8因素字典

    Returns:
        5个能力维度的详细信息
        {
            "专业力": {"score": 60, "level": "medium", "explanation": "..."},
            "管理力": {"score": 50, "level": "medium", "explanation": "..."},
            ...
        }
    """
    logger.info("[AbilityMapper] 开始映射HAY 8因素 → 5能力维度")

    # 提取各因素
    pk = hay_factors.get('practical_knowledge', 'D')
    mk = hay_factors.get('managerial_knowledge', 'I')
    comm = hay_factors.get('communication', '2')
    te = hay_factors.get('thinking_environment', 'D')
    tc = hay_factors.get('thinking_challenge', '3')
    fta = hay_factors.get('freedom_to_act', 'C')
    m = hay_factors.get('magnitude', 'N')
    ni = hay_factors.get('nature_of_impact', 'III')

    # 转换为分数
    pk_score = _get_level_score(pk, PK_SCORE_MAP)
    mk_score = _get_level_score(mk, MK_SCORE_MAP)
    comm_score = _get_level_score(comm, COMM_SCORE_MAP)
    te_score = _get_level_score(te, TE_SCORE_MAP)
    tc_score = _get_level_score(tc, TC_SCORE_MAP)
    fta_score = _get_level_score(fta, FTA_SCORE_MAP)
    ni_score = _get_ni_score(ni, m)

    abilities = {}

    # 1. 专业力 = PK(100%)
    professional_score = pk_score
    professional_level = _get_level_tag(professional_score)
    abilities["专业力"] = {
        "score": professional_score,
        "level": professional_level,
        "explanation": ABILITY_EXPLANATIONS["专业力"][professional_level]
    }

    # 2. 管理力 = MK(70%) + FTA(30%)
    management_score = _weighted_average([
        (mk_score, 0.7),
        (fta_score, 0.3)
    ])
    management_level = _get_level_tag(management_score)
    abilities["管理力"] = {
        "score": management_score,
        "level": management_level,
        "explanation": ABILITY_EXPLANATIONS["管理力"][management_level]
    }

    # 3. 合作力 = Comm(80%) + NI(20%)
    collaboration_score = _weighted_average([
        (comm_score, 0.8),
        (ni_score, 0.2)
    ])
    collaboration_level = _get_level_tag(collaboration_score)
    abilities["合作力"] = {
        "score": collaboration_score,
        "level": collaboration_level,
        "explanation": ABILITY_EXPLANATIONS["合作力"][collaboration_level]
    }

    # 4. 思辨力 = TE(100%)
    analytical_score = te_score
    analytical_level = _get_level_tag(analytical_score)
    abilities["思辨力"] = {
        "score": analytical_score,
        "level": analytical_level,
        "explanation": ABILITY_EXPLANATIONS["思辨力"][analytical_level]
    }

    # 5. 创新力 = TC(100%)
    innovation_score = tc_score
    innovation_level = _get_level_tag(innovation_score)
    abilities["创新力"] = {
        "score": innovation_score,
        "level": innovation_level,
        "explanation": ABILITY_EXPLANATIONS["创新力"][innovation_level]
    }

    logger.info(f"[AbilityMapper] 映射完成: 专业力={professional_score}, 管理力={management_score}, "
               f"合作力={collaboration_score}, 思辨力={analytical_score}, 创新力={innovation_score}")

    return abilities


# 向后兼容别名
map_factors_to_dimensions = map_hay_to_5_abilities


def get_ability_radar_data(abilities: Dict[str, Dict]) -> Dict[str, int]:
    """获取雷达图数据（只返回分数）"""
    return {name: info["score"] for name, info in abilities.items()}


# 向后兼容别名
get_dimension_radar_data = get_ability_radar_data


def get_ability_summary(abilities: Dict[str, Dict]) -> str:
    """生成能力总结文案"""
    sorted_abilities = sorted(abilities.items(), key=lambda x: x[1]["score"], reverse=True)
    highest = sorted_abilities[0]
    lowest = sorted_abilities[-1]

    summary = f"您的核心优势在于【{highest[0]}】，{highest[1]['explanation'][:30]}..."

    if lowest[1]["score"] < 50:
        summary += f" 建议关注【{lowest[0]}】的提升。"

    return summary


# 向后兼容别名
get_dimension_summary = get_ability_summary
