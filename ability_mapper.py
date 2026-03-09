"""
===========================================
能力映射模块 (Ability Mapper)
===========================================
将 HAY 8因素档位转换为 C端用户可理解的 8 个能力维度

8个能力维度（重命名以避免知识产权问题）：
1. 知识深度    ← PK  专业知识的深度与广度
2. 统筹能力    ← MK  组织协调与资源调度能力
3. 沟通影响    ← Comm 表达、说服与跨团队协作
4. 问题复杂度  ← TE  面对问题的模糊程度与复杂性
5. 创新思维    ← TC  突破常规、提出新方案的能力
6. 决策自主性  ← FTA 独立判断与自主决策的空间
7. 影响规模    ← M   工作成果影响的范围大小
8. 贡献类型    ← NI  对组织的贡献是辅助性还是主导性
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
# 8维能力 → 因素键名 映射
# ===========================================

# 因素键名 → (中文能力名, 分数映射表)
FACTOR_DIMENSION_MAP = {
    'practical_knowledge':  '知识深度',
    'managerial_knowledge': '统筹能力',
    'communication':        '沟通影响',
    'thinking_environment': '问题复杂度',
    'thinking_challenge':   '创新思维',
    'freedom_to_act':       '决策自主性',
    'magnitude':            '影响规模',
    'nature_of_impact':     '贡献类型',
}

# 每个维度的 high/medium/low 解释文案
DIMENSION_EXPLANATIONS = {
    "知识深度": {
        "high": "具备深厚的专业知识储备，能够处理高复杂度的专业问题",
        "medium": "具备扎实的专业基础，能够独立完成常规专业工作",
        "low": "具备基础专业知识，适合执行标准化的工作流程",
    },
    "统筹能力": {
        "high": "能够统筹多个业务模块，协调复杂的跨团队资源",
        "medium": "能够管理团队或项目，协调多方资源达成目标",
        "low": "能够管理自己的工作任务，配合团队完成目标",
    },
    "沟通影响": {
        "high": "能够影响关键决策，在复杂场景中建立广泛共识",
        "medium": "能够跨部门协作，有效沟通和推动项目落地",
        "low": "能够在团队内部有效沟通，配合完成协作任务",
    },
    "问题复杂度": {
        "high": "能够在高度模糊的环境中识别问题本质并做出判断",
        "medium": "能够分析复杂问题，在有一定框架的情境中独立思考",
        "low": "能够在清晰的框架下分析和解决问题",
    },
    "创新思维": {
        "high": "能够开拓新领域，提出突破性的解决方案",
        "medium": "能够改进现有流程和方法，推动持续优化",
        "low": "能够在现有框架下完成工作，学习和适应新方法",
    },
    "决策自主性": {
        "high": "能够在重大事项上独立决策，承担高层级判断责任",
        "medium": "能够在日常工作中自主决策，把握方向",
        "low": "在明确指引下执行工作，逐步积累判断经验",
    },
    "影响规模": {
        "high": "工作成果直接影响大规模业务或较大团队",
        "medium": "工作成果影响特定业务模块或项目范围",
        "low": "工作成果主要影响个人或小团队的产出",
    },
    "贡献类型": {
        "high": "在组织中承担核心产出角色，直接创造关键成果",
        "medium": "对业务成果有显著贡献，既有支持也有直接产出",
        "low": "以辅助和支持性贡献为主，协助推动业务目标",
    },
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


def map_factors_to_dimensions(hay_factors: Dict[str, str]) -> Dict[str, Dict]:
    """
    将 HAY 8因素转换为 8 个能力维度（1:1 直接映射，不做加权合并）

    Args:
        hay_factors: HAY 8因素字典
            {
                'practical_knowledge': 'E',
                'managerial_knowledge': 'II',
                'communication': '2',
                'thinking_environment': 'D',
                'thinking_challenge': '3',
                'freedom_to_act': 'C',
                'magnitude': '3',
                'nature_of_impact': 'S'
            }

    Returns:
        8个能力维度的详细信息
        {
            "知识深度": {"score": 60, "level": "medium", "grade": "E", "explanation": "..."},
            "统筹能力": {"score": 50, "level": "medium", "grade": "II", "explanation": "..."},
            ...
        }
    """
    logger.info("[AbilityMapper] 开始映射 HAY 8因素 → 8 能力维度")

    # 各因素对应的分数映射表
    score_maps = {
        'practical_knowledge':  PK_SCORE_MAP,
        'managerial_knowledge': MK_SCORE_MAP,
        'communication':        COMM_SCORE_MAP,
        'thinking_environment': TE_SCORE_MAP,
        'thinking_challenge':   TC_SCORE_MAP,
        'freedom_to_act':       FTA_SCORE_MAP,
        'magnitude':            M_SCORE_MAP,
    }

    dimensions = {}

    for factor_key, dimension_name in FACTOR_DIMENSION_MAP.items():
        grade = hay_factors.get(factor_key, '')

        # NI 需要特殊处理（依赖 M 的值来选择评分系统）
        if factor_key == 'nature_of_impact':
            m_value = hay_factors.get('magnitude', 'N')
            score = _get_ni_score(grade, m_value)
        else:
            score_map = score_maps[factor_key]
            score = _get_level_score(grade, score_map)

        level = _get_level_tag(score)

        dimensions[dimension_name] = {
            "score": score,
            "level": level,
            "grade": grade,
            "explanation": DIMENSION_EXPLANATIONS[dimension_name][level],
        }

    log_parts = [f"{name}={info['score']}" for name, info in dimensions.items()]
    logger.info(f"[AbilityMapper] 映射完成: {', '.join(log_parts)}")

    return dimensions


def get_dimension_radar_data(dimensions: Dict[str, Dict]) -> Dict[str, int]:
    """获取雷达图数据（只返回分数）"""
    return {name: info["score"] for name, info in dimensions.items()}


def get_dimension_summary(dimensions: Dict[str, Dict]) -> str:
    """生成能力总结文案"""
    sorted_dims = sorted(dimensions.items(), key=lambda x: x[1]["score"], reverse=True)
    highest = sorted_dims[0]
    lowest = sorted_dims[-1]

    summary = f"您的核心优势在于【{highest[0]}】，{highest[1]['explanation'][:30]}..."

    if lowest[1]["score"] < 50:
        summary += f" 建议关注【{lowest[0]}】的提升。"

    return summary


# ===========================================
# 向后兼容（保留旧函数名，内部转发）
# ===========================================

def map_hay_to_5_abilities(hay_factors: Dict[str, str]) -> Dict[str, Dict]:
    """向后兼容: 转发到 map_factors_to_dimensions"""
    return map_factors_to_dimensions(hay_factors)


def get_ability_radar_data(abilities: Dict[str, Dict]) -> Dict[str, int]:
    """向后兼容: 转发到 get_dimension_radar_data"""
    return get_dimension_radar_data(abilities)


def get_ability_summary(abilities: Dict[str, Dict]) -> str:
    """向后兼容: 转发到 get_dimension_summary"""
    return get_dimension_summary(abilities)
