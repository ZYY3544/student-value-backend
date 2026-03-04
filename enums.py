"""
===========================================
枚举类型定义
===========================================

定义HAY评估系统中所有因素的有效值（枚举）
用于Pydantic模型验证，确保LLM返回的值在有效范围内
"""

from enum import Enum


# ===========================================
# Know-How 维度枚举
# ===========================================

class PracticalKnowledge(str, Enum):
    """实践经验/专业领域知识"""
    A_MINUS = "A-"
    A = "A"
    A_PLUS = "A+"
    B_MINUS = "B-"
    B = "B"
    B_PLUS = "B+"
    C_MINUS = "C-"
    C = "C"
    C_PLUS = "C+"
    D_MINUS = "D-"
    D = "D"
    D_PLUS = "D+"
    E_MINUS = "E-"
    E = "E"
    E_PLUS = "E+"
    F_MINUS = "F-"
    F = "F"
    F_PLUS = "F+"
    G_MINUS = "G-"
    G = "G"
    G_PLUS = "G+"
    H_MINUS = "H-"
    H = "H"
    H_PLUS = "H+"
    I_MINUS = "I-"
    I = "I"
    I_PLUS = "I+"


class ManagerialKnowledge(str, Enum):
    """计划、组织与整合知识"""
    T_MINUS = "T-"
    T = "T"
    T_PLUS = "T+"
    I_MINUS = "I-"
    I = "I"
    I_PLUS = "I+"
    II_MINUS = "II-"
    II = "II"
    II_PLUS = "II+"
    III_MINUS = "III-"
    III = "III"
    III_PLUS = "III+"
    IV_MINUS = "IV-"
    IV = "IV"
    IV_PLUS = "IV+"
    V_MINUS = "V-"
    V = "V"
    V_PLUS = "V+"
    VI_MINUS = "VI-"
    VI = "VI"
    VI_PLUS = "VI+"
    VII_MINUS = "VII-"
    VII = "VII"
    VII_PLUS = "VII+"
    VIII_MINUS = "VIII-"
    VIII = "VIII"
    VIII_PLUS = "VIII+"
    IX_MINUS = "IX-"
    IX = "IX"
    IX_PLUS = "IX+"


class Communication(str, Enum):
    """沟通与影响技能"""
    LEVEL_1 = "1"
    LEVEL_2 = "2"
    LEVEL_3 = "3"


# ===========================================
# Problem Solving 维度枚举
# ===========================================

class ThinkingChallenge(str, Enum):
    """思考挑战"""
    LEVEL_1_MINUS = "1-"
    LEVEL_1 = "1"
    LEVEL_1_PLUS = "1+"
    LEVEL_2_MINUS = "2-"
    LEVEL_2 = "2"
    LEVEL_2_PLUS = "2+"
    LEVEL_3_MINUS = "3-"
    LEVEL_3 = "3"
    LEVEL_3_PLUS = "3+"
    LEVEL_4_MINUS = "4-"
    LEVEL_4 = "4"
    LEVEL_4_PLUS = "4+"
    LEVEL_5_MINUS = "5-"
    LEVEL_5 = "5"
    LEVEL_5_PLUS = "5+"


class ThinkingEnvironment(str, Enum):
    """思考环境"""
    A_MINUS = "A-"
    A = "A"
    A_PLUS = "A+"
    B_MINUS = "B-"
    B = "B"
    B_PLUS = "B+"
    C_MINUS = "C-"
    C = "C"
    C_PLUS = "C+"
    D_MINUS = "D-"
    D = "D"
    D_PLUS = "D+"
    E_MINUS = "E-"
    E = "E"
    E_PLUS = "E+"
    F_MINUS = "F-"
    F = "F"
    F_PLUS = "F+"
    G_MINUS = "G-"
    G = "G"
    G_PLUS = "G+"
    H_MINUS = "H-"
    H = "H"
    H_PLUS = "H+"


# ===========================================
# Accountability 维度枚举
# ===========================================

class FreedomToAct(str, Enum):
    """行动自由度"""
    A_MINUS = "A-"
    A = "A"
    A_PLUS = "A+"
    B_MINUS = "B-"
    B = "B"
    B_PLUS = "B+"
    C_MINUS = "C-"
    C = "C"
    C_PLUS = "C+"
    D_MINUS = "D-"
    D = "D"
    D_PLUS = "D+"
    E_MINUS = "E-"
    E = "E"
    E_PLUS = "E+"
    F_MINUS = "F-"
    F = "F"
    F_PLUS = "F+"
    G_MINUS = "G-"
    G = "G"
    G_PLUS = "G+"
    H_MINUS = "H-"
    H = "H"
    H_PLUS = "H+"
    I_MINUS = "I-"
    I = "I"
    I_PLUS = "I+"


class Magnitude(str, Enum):
    """影响范围（可量化模式）"""
    N = "N"  # 不可量化
    LEVEL_1_MINUS = "1-"
    LEVEL_1 = "1"
    LEVEL_1_PLUS = "1+"
    LEVEL_2_MINUS = "2-"
    LEVEL_2 = "2"
    LEVEL_2_PLUS = "2+"
    LEVEL_3_MINUS = "3-"
    LEVEL_3 = "3"
    LEVEL_3_PLUS = "3+"
    LEVEL_4_MINUS = "4-"
    LEVEL_4 = "4"
    LEVEL_4_PLUS = "4+"
    LEVEL_5_MINUS = "5-"
    LEVEL_5 = "5"
    LEVEL_5_PLUS = "5+"


class NatureOfImpactQuantified(str, Enum):
    """影响性质（可量化模式）"""
    R_MINUS = "R-"
    R = "R"
    R_PLUS = "R+"
    C_MINUS = "C-"
    C = "C"
    C_PLUS = "C+"
    S_MINUS = "S-"
    S = "S"
    S_PLUS = "S+"
    P_MINUS = "P-"
    P = "P"
    P_PLUS = "P+"


class NatureOfImpactNonQuantified(str, Enum):
    """影响性质（不可量化模式）"""
    I_MINUS = "I-"
    I = "I"
    I_PLUS = "I+"
    II_MINUS = "II-"
    II = "II"
    II_PLUS = "II+"
    III_MINUS = "III-"
    III = "III"
    III_PLUS = "III+"
    IV_MINUS = "IV-"
    IV = "IV"
    IV_PLUS = "IV+"
    V_MINUS = "V-"
    V = "V"
    V_PLUS = "V+"
    VI_MINUS = "VI-"
    VI = "VI"
    VI_PLUS = "VI+"


# ===========================================
# 辅助函数
# ===========================================

def get_all_enum_values(enum_class) -> list[str]:
    """获取枚举类的所有值"""
    return [e.value for e in enum_class]


def is_valid_factor_value(factor_name: str, value: str) -> bool:
    """
    检查因素值是否有效

    Args:
        factor_name: 因素名称
        value: 因素值

    Returns:
        是否有效
    """
    enum_mapping = {
        'practical_knowledge': PracticalKnowledge,
        'managerial_knowledge': ManagerialKnowledge,
        'communication': Communication,
        'thinking_challenge': ThinkingChallenge,
        'thinking_environment': ThinkingEnvironment,
        'freedom_to_act': FreedomToAct,
        'magnitude': Magnitude,
    }

    enum_class = enum_mapping.get(factor_name)
    if not enum_class:
        return False

    return value in get_all_enum_values(enum_class)
