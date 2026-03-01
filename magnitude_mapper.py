"""
===========================================
Magnitude 映射模块
===========================================

作用：根据营收金额自动映射到对应的Magnitude档位（带+/-后缀）

金额范围（人民币）：
- 1档: 30万-300万
- 2档: 300万-3000万
- 3档: 3000万-3亿
- 4档: 3亿-30亿
- 5档: 30亿-300亿
- 6档: 300亿-3000亿
- 7档: 3000亿-3万亿
- 8档: 3万亿-30万亿
- 9档: 30万亿以上

+/-后缀规则（对数均分）：
- 每个档位区间 [L, U] (U = 10×L) 对数均分为三段：
  * 档位"-": [L, L × 10^(1/3)] ≈ [L, L × 2.15]
  * 档位无后缀: [L × 10^(1/3), L × 10^(2/3)] ≈ [L × 2.15, L × 4.64]
  * 档位"+": [L × 10^(2/3), U] ≈ [L × 4.64, L × 10]
"""

import math


def map_amount_to_magnitude(amount_wan: float) -> str:
    """
    将金额（万元）映射到对应的Magnitude档位（带+/-后缀）

    使用对数均分方法，将每个档位的10倍区间分为三段

    参数:
        amount_wan: 金额，单位为万元

    返回:
        str: Magnitude档位，如 '1-', '1', '1+', '2-', '2', '2+', ... '9+'
    """
    # 档位定义：每个档位的下限（万元）
    # 1档: 30万, 2档: 300万, 3档: 3000万, ...
    magnitude_ranges = [
        (30, '1'),        # 1档: 30万-300万
        (300, '2'),       # 2档: 300万-3000万
        (3000, '3'),      # 3档: 3000万-3亿
        (30000, '4'),     # 4档: 3亿-30亿
        (300000, '5'),    # 5档: 30亿-300亿
        (3000000, '6'),   # 6档: 300亿-3000亿
        (30000000, '7'),  # 7档: 3000亿-3万亿
        (300000000, '8'), # 8档: 3万亿-30万亿
        (3000000000, '9') # 9档: 30万亿以上
    ]

    # 特殊处理：小于30万
    if amount_wan < 30:
        return '1-'

    # 找到对应的档位
    base_level = None
    lower_bound = None

    for i in range(len(magnitude_ranges)):
        lower_bound = magnitude_ranges[i][0]
        base_level = magnitude_ranges[i][1]

        # 检查是否在当前档位范围内
        if i < len(magnitude_ranges) - 1:
            upper_bound = magnitude_ranges[i + 1][0]
            if amount_wan < upper_bound:
                break
        else:
            # 最高档位（9档）
            break

    # 对数均分：计算+/-后缀
    # 分界点1: L × 10^(1/3) ≈ L × 2.154
    # 分界点2: L × 10^(2/3) ≈ L × 4.642
    threshold_minus_to_base = lower_bound * math.pow(10, 1/3)  # 约 L × 2.154
    threshold_base_to_plus = lower_bound * math.pow(10, 2/3)   # 约 L × 4.642

    if amount_wan <= threshold_minus_to_base:
        return f"{base_level}-"
    elif amount_wan <= threshold_base_to_plus:
        return base_level
    else:
        return f"{base_level}+"


def get_magnitude_description(magnitude: str) -> str:
    """
    获取Magnitude档位的描述

    参数:
        magnitude: 档位代码，如 '1-', '1', '1+', '2', '4+', 'N'

    返回:
        str: 档位描述
    """
    # 提取基础档位（去掉+/-）
    base_level = magnitude.replace('-', '').replace('+', '')

    base_descriptions = {
        '1': '极小规模 (30万-300万)',
        '2': '小规模 (300万-3000万)',
        '3': '中等规模 (3000万-3亿)',
        '4': '大规模 (3亿-30亿)',
        '5': '极大规模 (30亿-300亿)',
        '6': '超大规模 (300亿-3000亿)',
        '7': '巨大规模 (3000亿-3万亿)',
        '8': '超巨规模 (3万亿-30万亿)',
        '9': '顶级规模 (30万亿以上)',
        'N': '无显著影响（不可量化）'
    }

    base_desc = base_descriptions.get(base_level, '未知档位')

    # 添加+/-说明
    if magnitude.endswith('-'):
        return f"{base_desc} [偏低端]"
    elif magnitude.endswith('+'):
        return f"{base_desc} [偏高端]"
    else:
        return f"{base_desc} [中等]"


if __name__ == '__main__':
    # 测试：对数均分验证
    test_cases = [
        # 4档测试（3亿-30亿）
        (30000, '4-'),      # 3亿 → 4-
        (40000, '4-'),      # 4亿 → 4-（< 6.46亿）
        (60000, '4-'),      # 6亿 → 4-
        (70000, '4'),       # 7亿 → 4（> 6.46亿）
        (100000, '4'),      # 10亿 → 4
        (130000, '4'),      # 13亿 → 4（< 13.93亿）
        (150000, '4+'),     # 15亿 → 4+（> 13.93亿）
        (250000, '4+'),     # 25亿 → 4+
        (290000, '4+'),     # 29亿 → 4+

        # 其他档位测试
        (50, '1-'),         # 50万 → 1-
        (100, '1'),         # 100万 → 1
        (250, '1+'),        # 250万 → 1+
        (500, '2-'),        # 500万 → 2-
        (5000, '3-'),       # 5000万 → 3-
        (500000, '5-'),     # 50亿 → 5-
    ]

    print("Magnitude 对数均分映射测试")
    print("=" * 80)
    print(f"{'金额(万元)':<15} {'实际结果':<10} {'预期结果':<10} {'描述':<40} {'状态'}")
    print("=" * 80)

    for amount, expected in test_cases:
        result = map_amount_to_magnitude(amount)
        desc = get_magnitude_description(result)
        status = "✅" if result == expected else f"❌ (得到{result})"

        # 格式化金额显示
        if amount >= 10000:
            amount_str = f"{amount/10000:.1f}亿"
        else:
            amount_str = f"{amount}万"

        print(f"{amount_str:<15} {result:<10} {expected:<10} {desc:<40} {status}")

    print("=" * 80)

    # 显示对数均分的分界点
    print("\n对数均分分界点参考：")
    print("=" * 80)
    for level, lower in [(4, 30000), (5, 300000)]:
        threshold1 = lower * math.pow(10, 1/3)
        threshold2 = lower * math.pow(10, 2/3)
        print(f"{level}档：{lower/10000:.0f}亿 - {lower*10/10000:.0f}亿")
        print(f"  {level}-: {lower/10000:.2f}亿 - {threshold1/10000:.2f}亿")
        print(f"  {level} : {threshold1/10000:.2f}亿 - {threshold2/10000:.2f}亿")
        print(f"  {level}+: {threshold2/10000:.2f}亿 - {lower*10/10000:.0f}亿")
    print("=" * 80)
