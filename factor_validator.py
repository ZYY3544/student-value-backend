"""
HAY 因素组合验证器
Orchestrates 5-layer validation with detailed feedback

验证层说明：
1. Know-How (KH) - 知识技能三因素内部验证
2. Problem Solving (PS) - 解决问题两因素内部验证
3. Accountability (ACC) - 职责三因素内部验证
4. PS×KH - 解决问题与知识技能跨维度匹配验证
5. 跨因素等级约束 - 专业知识 >= 思考环境 >= 行动自由度
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from models import HayFactors
from validation_rules import validation_rules
from data_tables import (
    PRACTICAL_KNOWLEDGE_SCORES,
    MANAGERIAL_KNOWLEDGE_SCORES,
    COMMUNICATION_SCORES,
    THINKING_ENVIRONMENT_SCORES,
    THINKING_CHALLENGE_SCORES,
    PS_PERCENTAGE_TABLE,
    LEVEL_SCORE_TABLE
)


@dataclass
class ValidationLayerResult:
    """单层验证结果"""
    layer_name: str  # 层名称: "KH", "PS", "ACC", "PS×KH"
    is_valid: bool  # 是否通过验证
    result_label: str  # 结果标签: "正常", "有可能出现但不建议", "错误"
    probability: str  # 概率: "100%", "50%", "0%"
    factors_checked: Dict[str, str]  # 检查的因素及其值
    error_message: Optional[str] = None  # 错误详情


@dataclass
class FullValidationResult:
    """完整的五层验证结果"""
    all_passed: bool  # 是否全部通过（严格模式：100%）
    layers: List[ValidationLayerResult]  # 各层结果（5层）
    overall_message: str  # 总体消息

    def get_failed_layers(self) -> List[ValidationLayerResult]:
        """获取所有未通过的层"""
        return [layer for layer in self.layers if not layer.is_valid]

    def has_errors(self) -> bool:
        """是否有错误（0%概率）"""
        return any(layer.probability == "0%" for layer in self.layers if not layer.is_valid)

    def has_warnings(self) -> bool:
        """是否有警告（50%概率）"""
        return any(layer.probability == "50%" for layer in self.layers if not layer.is_valid)


class FactorValidator:
    """HAY 因素验证器"""

    def __init__(self):
        """初始化验证器"""
        self.rules = validation_rules

    @staticmethod
    def _parse_factor_level(factor_value: str) -> float:
        """
        将因素等级字符串转换为可比较的数值

        例如：A- = 0.7, A = 1.0, A+ = 1.3, B- = 1.7, B = 2.0, B+ = 2.3 ...

        Args:
            factor_value: 等级字符串 (如 'A', 'B+', 'E-')

        Returns:
            可比较的数值
        """
        if not factor_value or factor_value == 'N':
            return 0.0

        # 提取基础字母（A, B, C, D, E, F, G, H）
        base_letter = factor_value[0]

        # 字母到数值的映射
        letter_values = {
            'A': 1, 'B': 2, 'C': 3, 'D': 4,
            'E': 5, 'F': 6, 'G': 7, 'H': 8
        }

        base_value = letter_values.get(base_letter, 0)

        # 处理符号
        if factor_value.endswith('+'):
            return base_value + 0.3
        elif factor_value.endswith('-'):
            return base_value - 0.3
        else:
            return float(base_value)

    def validate_all_layers(self, factors: HayFactors) -> FullValidationResult:
        """
        执行五层验证

        Args:
            factors: HAY 八因素对象

        Returns:
            完整的验证结果
        """
        layers = []

        # Layer 1: Know-How 内部验证
        kh_result = self._validate_kh_layer(factors)
        layers.append(kh_result)

        # Layer 2: Problem Solving 内部验证
        ps_result = self._validate_ps_layer(factors)
        layers.append(ps_result)

        # Layer 3: Accountability 内部验证
        acc_result = self._validate_acc_layer(factors)
        layers.append(acc_result)

        # Layer 4: PS×KH 跨维度验证 (需要先计算分数)
        ps_kh_result = self._validate_ps_kh_layer(factors)
        layers.append(ps_kh_result)

        # Layer 5: 跨因素等级约束验证（专业知识 >= 思考环境 >= 行动自由度）
        hierarchy_result = self._validate_cross_factor_hierarchy(factors)
        layers.append(hierarchy_result)

        # 判断是否全部通过（严格模式）
        all_passed = all(layer.is_valid for layer in layers)

        # 调试信息：打印每层验证结果
        print(f"\n[DEBUG] 验证结果详情:")
        for i, layer in enumerate(layers, 1):
            status = "PASS" if layer.is_valid else "FAIL"
            print(f"  第{i}层 ({layer.layer_name}): {status} - {layer.result_label}")
            if not layer.is_valid and layer.error_message:
                print(f"    错误: {layer.error_message}")

        # 生成总体消息
        if all_passed:
            overall_message = "[PASS] 所有因素组合验证通过"
        else:
            failed_count = len([l for l in layers if not l.is_valid])
            overall_message = f"[FAIL] {failed_count}/5 层验证未通过"

        return FullValidationResult(
            all_passed=all_passed,
            layers=layers,
            overall_message=overall_message
        )

    def _validate_kh_layer(self, factors: HayFactors) -> ValidationLayerResult:
        """验证 Know-How 层"""
        practical = factors.practical_knowledge
        managerial = factors.managerial_knowledge
        communication = factors.communication

        is_valid, result_label, probability = self.rules.validate_kh(
            practical, managerial, communication
        )

        error_msg = None
        if not is_valid:
            error_msg = (
                f"Know-How 组合不合理: "
                f"实践经验={practical}, 管理知识={managerial}, 沟通技巧={communication} "
                f"({result_label}, 概率={probability})"
            )

        return ValidationLayerResult(
            layer_name="Know-How (KH)",
            is_valid=is_valid,
            result_label=result_label,
            probability=probability,
            factors_checked={
                "实践经验/专业知识": practical,
                "管理知识": managerial,
                "沟通技巧": communication
            },
            error_message=error_msg
        )

    def _validate_ps_layer(self, factors: HayFactors) -> ValidationLayerResult:
        """验证 Problem Solving 层"""
        environment = factors.thinking_environment
        challenge = factors.thinking_challenge

        is_valid, result_label, probability = self.rules.validate_ps(
            environment, challenge
        )

        error_msg = None
        if not is_valid:
            error_msg = (
                f"Problem Solving 组合不合理: "
                f"思维环境={environment}, 思维挑战={challenge} "
                f"({result_label}, 概率={probability})"
            )

        return ValidationLayerResult(
            layer_name="Problem Solving (PS)",
            is_valid=is_valid,
            result_label=result_label,
            probability=probability,
            factors_checked={
                "思维环境": environment,
                "思维挑战": challenge
            },
            error_message=error_msg
        )

    def _validate_acc_layer(self, factors: HayFactors) -> ValidationLayerResult:
        """验证 Accountability 层（特殊逻辑：magnitude='N'）"""
        freedom = factors.freedom_to_act
        magnitude = factors.magnitude
        nature = factors.nature_of_impact

        is_valid, result_label, probability = self.rules.validate_acc(
            freedom, magnitude, nature
        )

        error_msg = None
        if not is_valid:
            error_msg = (
                f"Accountability 组合不合理: "
                f"行动自由度={freedom}, 影响范围={magnitude}, 影响性质={nature} "
                f"({result_label}, 概率={probability})"
            )

        return ValidationLayerResult(
            layer_name="Accountability (ACC)",
            is_valid=is_valid,
            result_label=result_label,
            probability=probability,
            factors_checked={
                "行动自由度": freedom,
                "影响范围": magnitude,
                "影响性质": nature
            },
            error_message=error_msg
        )

    def _validate_cross_factor_hierarchy(self, factors: HayFactors) -> ValidationLayerResult:
        """
        验证跨因素等级约束（专业知识 >= 思考环境 >= 行动自由度）

        约束规则：
        1. 专业知识技能 >= 思考环境 (大脑容量 >= 消耗量)
        2. 专业知识技能 >= 行动自由度
        3. 思考环境 >= 行动自由度

        Returns:
            ValidationLayerResult: 验证结果
        """
        practical = factors.practical_knowledge
        environment = factors.thinking_environment
        freedom = factors.freedom_to_act

        # 转换为可比较的数值
        practical_level = self._parse_factor_level(practical)
        environment_level = self._parse_factor_level(environment)
        freedom_level = self._parse_factor_level(freedom)

        # 检查三个约束
        errors = []

        # 约束1：专业知识 >= 思考环境
        if practical_level < environment_level:
            errors.append(
                f"专业知识技能({practical}={practical_level:.1f}) < 思考环境({environment}={environment_level:.1f})"
            )

        # 约束2：专业知识 >= 行动自由度
        if practical_level < freedom_level:
            errors.append(
                f"专业知识技能({practical}={practical_level:.1f}) < 行动自由度({freedom}={freedom_level:.1f})"
            )

        # 约束3：思考环境 >= 行动自由度
        if environment_level < freedom_level:
            errors.append(
                f"思考环境({environment}={environment_level:.1f}) < 行动自由度({freedom}={freedom_level:.1f})"
            )

        # 判断验证结果
        is_valid = len(errors) == 0

        if is_valid:
            result_label = "正常"
            probability = "100%"
            error_msg = None
        else:
            result_label = "错误"
            probability = "0%"
            error_msg = "因素等级约束违反:\n  " + "\n  ".join(errors) + \
                       "\n\n约束规则: 专业知识技能 >= 思考环境 >= 行动自由度"

        return ValidationLayerResult(
            layer_name="跨因素等级约束",
            is_valid=is_valid,
            result_label=result_label,
            probability=probability,
            factors_checked={
                "专业知识技能": practical,
                "思考环境": environment,
                "行动自由度": freedom
            },
            error_message=error_msg
        )

    def _validate_ps_kh_layer(self, factors: HayFactors) -> ValidationLayerResult:
        """验证 PS×KH 跨维度匹配"""
        # 计算 KH 分数
        kh_score = self._calculate_kh_score(factors)

        # 获取 PS 百分比
        ps_percentage = self._get_ps_percentage(factors)

        is_valid, result_label, probability = self.rules.validate_ps_kh_cross(
            ps_percentage, kh_score
        )

        error_msg = None
        if not is_valid:
            error_msg = (
                f"PS×KH 跨维度匹配不合理: "
                f"PS百分比={ps_percentage}, KH分数={kh_score} "
                f"({result_label}, 概率={probability})"
            )

        return ValidationLayerResult(
            layer_name="PS×KH 跨维度",
            is_valid=is_valid,
            result_label=result_label,
            probability=probability,
            factors_checked={
                "PS百分比": ps_percentage,
                "KH分数": str(kh_score)
            },
            error_message=error_msg
        )

    def _calculate_kh_score(self, factors: HayFactors) -> int:
        """
        计算 KH 最终分数（使用与calculator.py相同的逻辑）

        Returns:
            最终的KH分数（经过LEVEL_SCORE_TABLE转换）
        """
        # 1. 获取各因素的基础分数
        practical_score = PRACTICAL_KNOWLEDGE_SCORES.get(factors.practical_knowledge, 0)
        managerial_score = MANAGERIAL_KNOWLEDGE_SCORES.get(factors.managerial_knowledge, 0)
        communication_score = COMMUNICATION_SCORES.get(factors.communication, 0)

        # 2. 计算base level
        base_level = practical_score + managerial_score + communication_score

        # 3. 计算符号调整
        symbol_adjustment = self._calculate_kh_symbol_adjustment(
            factors.practical_knowledge,
            factors.managerial_knowledge
        )

        # 4. 得到最终level
        kh_level = base_level + symbol_adjustment

        # 5. 转换为最终分数
        kh_score = LEVEL_SCORE_TABLE.get(kh_level, 0)

        return kh_score

    @staticmethod
    def _calculate_kh_symbol_adjustment(practical: str, managerial: str) -> int:
        """
        计算 Know How 的符号调整

        修复P2-9：使用utils.py的统一函数，避免逻辑重复
        """
        from utils import calculate_kh_symbol_adjustment
        return calculate_kh_symbol_adjustment(practical, managerial)

    def _get_ps_percentage(self, factors: HayFactors) -> str:
        """
        获取 PS 百分比字符串
        使用与 calculator.py 相同的逻辑（包含符号调整）
        """
        c_score = THINKING_CHALLENGE_SCORES.get(factors.thinking_challenge, 0)
        e_score = THINKING_ENVIRONMENT_SCORES.get(factors.thinking_environment, 0)

        # 计算基础等级
        base_level = c_score + e_score

        # 计算符号调整（与calculator.py保持一致）
        symbol_adjustment = self._calculate_ps_symbol_adjustment(
            factors.thinking_challenge,
            factors.thinking_environment
        )

        # 最终PS Base Level
        ps_base_level = base_level + symbol_adjustment

        # 从百分比表获取百分比
        ps_percentage_decimal = PS_PERCENTAGE_TABLE.get(ps_base_level, 0.0)

        # 转换为百分比字符串
        return f"{int(ps_percentage_decimal * 100)}%"

    @staticmethod
    def _calculate_ps_symbol_adjustment(challenge: str, environment: str) -> int:
        """
        计算 Problem Solving 的符号调整

        修复P2-9：使用utils.py的统一函数，避免逻辑重复
        """
        from utils import calculate_ps_symbol_adjustment
        return calculate_ps_symbol_adjustment(challenge, environment)


# 全局单例
factor_validator = FactorValidator()
