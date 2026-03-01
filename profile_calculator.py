"""
===========================================
岗位特性计算模块 (Job Profile Calculator)
===========================================

作用：根据PS Level和ACC Level计算岗位特性(Short Profile)

岗位特性定义：
- P型岗位：解决问题主导（PS Level > ACC Level）
- L型岗位：平衡型（PS Level = ACC Level）
- A型岗位：责任性主导（ACC Level > PS Level）
"""

from typing import Tuple


class ProfileCalculator:
    """
    岗位特性计算器

    根据PS Level和ACC Level的差距计算岗位特性（P4/P3/P2/P1/L/A1/A2/A3/A4）
    """

    # 岗位特性直接映射规则（基于Level差距）
    # 直接映射：Gap值 = Profile类型数字
    # Gap >= 4 → P4, Gap = 3 → P3, ..., Gap = 0 → L, Gap = -1 → A1, ..., Gap <= -4 → A4

    @staticmethod
    def calculate_profile(ps_level: int, acc_level: int) -> Tuple[str, int, str]:
        """
        计算岗位特性

        Args:
            ps_level: PS维度的Level
            acc_level: ACC维度的Level

        Returns:
            (profile_type, level_gap, description)
            - profile_type: 岗位特性类型 (P4/P3/P2/P1/L/A1/A2/A3/A4)
            - level_gap: Level差距 (正数表示PS主导，负数表示ACC主导)
            - description: 岗位特性描述
        """
        level_gap = ps_level - acc_level

        # 根据差距确定岗位特性类型
        profile_type = ProfileCalculator._get_profile_type(level_gap)

        # 生成描述
        description = ProfileCalculator._get_profile_description(profile_type, level_gap)

        return profile_type, level_gap, description

    @staticmethod
    def _get_profile_type(level_gap: int) -> str:
        """
        根据Level差距确定岗位特性类型（直接映射）

        规则：
        - Gap >= 4 → P4
        - Gap = 3 → P3
        - Gap = 2 → P2
        - Gap = 1 → P1
        - Gap = 0 → L
        - Gap = -1 → A1
        - Gap = -2 → A2
        - Gap = -3 → A3
        - Gap <= -4 → A4
        """
        if level_gap >= 4:
            return 'P4'
        elif level_gap == 3:
            return 'P3'
        elif level_gap == 2:
            return 'P2'
        elif level_gap == 1:
            return 'P1'
        elif level_gap == 0:
            return 'L'
        elif level_gap == -1:
            return 'A1'
        elif level_gap == -2:
            return 'A2'
        elif level_gap == -3:
            return 'A3'
        else:  # level_gap <= -4
            return 'A4'

    @staticmethod
    def _get_profile_description(profile_type: str, level_gap: int) -> str:
        """生成岗位特性描述"""
        descriptions = {
            'P4': f"强解决问题型岗位（PS超出ACC {level_gap}级）",
            'P3': f"解决问题主导岗位（PS超出ACC {level_gap}级）",
            'P2': f"偏重解决问题岗位（PS超出ACC {level_gap}级）",
            'P1': f"轻微偏重解决问题（PS超出ACC {level_gap}级）",
            'L': "平衡型岗位（PS与ACC相当）",
            'A1': f"轻微偏重责任性（ACC超出PS {abs(level_gap)}级）",
            'A2': f"偏重责任性岗位（ACC超出PS {abs(level_gap)}级）",
            'A3': f"责任性主导岗位（ACC超出PS {abs(level_gap)}级）",
            'A4': f"强责任性型岗位（ACC超出PS {abs(level_gap)}级）"
        }
        return descriptions.get(profile_type, "未知类型")

    @staticmethod
    def get_profile_category(profile_type: str) -> str:
        """获取岗位特性的大类（P型/L型/A型）"""
        if profile_type.startswith('P'):
            return 'P型（解决问题主导）'
        elif profile_type == 'L':
            return 'L型（平衡型）'
        elif profile_type.startswith('A'):
            return 'A型（责任性主导）'
        else:
            return '未知类型'


def calculate_job_profile(ps_level: int, acc_level: int) -> dict:
    """
    计算岗位特性的便捷函数

    Args:
        ps_level: PS维度的Level
        acc_level: ACC维度的Level

    Returns:
        {
            'profile_type': 'A2',
            'profile_category': 'A型（责任性主导）',
            'level_gap': -7,
            'description': '偏重责任性岗位（ACC超出PS 7级）',
            'ps_level': 15,
            'acc_level': 22
        }
    """
    profile_type, level_gap, description = ProfileCalculator.calculate_profile(ps_level, acc_level)
    profile_category = ProfileCalculator.get_profile_category(profile_type)

    return {
        'profile_type': profile_type,
        'profile_category': profile_category,
        'level_gap': level_gap,
        'description': description,
        'ps_level': ps_level,
        'acc_level': acc_level
    }


# 示例用法
if __name__ == '__main__':
    # 测试案例
    test_cases = [
        (10, 10, "L型 - 平衡"),
        (15, 10, "P型 - PS主导"),
        (10, 15, "A型 - ACC主导"),
        (20, 10, "P3型 - 强PS主导"),
        (10, 22, "A3型 - 强ACC主导")
    ]

    print("岗位特性计算测试：\n")
    for ps_level, acc_level, expected in test_cases:
        result = calculate_job_profile(ps_level, acc_level)
        print(f"PS={ps_level}, ACC={acc_level} ({expected})")
        print(f"  → {result['profile_type']} - {result['description']}")
        print()
