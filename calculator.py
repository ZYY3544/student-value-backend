"""
===========================================
核心计算模块 (Calculator Module) - Refactored
===========================================

作用：实现 Hay 三维度八因素的完整计算逻辑

改进点：
1. 使用 Pydantic 模型进行输入验证
2. 强类型返回值
3. 更清晰的逻辑结构
"""

from data_tables import (
    PRACTICAL_KNOWLEDGE_SCORES, MANAGERIAL_KNOWLEDGE_SCORES, COMMUNICATION_SCORES,
    THINKING_CHALLENGE_SCORES, THINKING_ENVIRONMENT_SCORES,
    FREEDOM_TO_ACT_SCORES, MAGNITUDE_SCORES_QUANTIFIED,
    NATURE_OF_IMPACT_SCORES_QUANTIFIED, NATURE_OF_IMPACT_SCORES_NON_QUANTIFIED,
    LEVEL_SCORE_TABLE, PS_PERCENTAGE_TABLE, PS_MATRIX,
    get_job_grade, get_level_score, get_ps_level_from_matrix
)
from models import (
    HayFactors, HayResult, HaySummary, JobProfile,
    KnowHowResult, ProblemSolvingResult, AccountabilityResult
)
from profile_calculator import calculate_job_profile

class HayCalculator:
    """
    Hay 评估计算器 (Type-Safe Version)
    """

    @staticmethod
    def _extract_symbol(factor_value: str) -> str:
        """提取因素级别的符号 (+, -, N)"""
        if factor_value.endswith('+'):
            return '+'
        elif factor_value.endswith('-'):
            return '-'
        else:
            return 'N'

    @staticmethod
    def _calculate_kh_symbol_adjustment(practical: str, managerial: str) -> int:
        """计算 Know How 的符号调整"""
        p_sym = HayCalculator._extract_symbol(practical)
        m_sym = HayCalculator._extract_symbol(managerial)
        
        symbols = [p_sym, m_sym]
        plus_count = symbols.count('+')
        minus_count = symbols.count('-')

        if plus_count == 2 or (plus_count == 1 and minus_count == 0):
            return 2
        elif (plus_count == 0 and minus_count == 0) or (plus_count == 1 and minus_count == 1):
            return 1
        else:
            return 0

    @staticmethod
    def _calculate_ps_symbol_adjustment(challenge: str, environment: str) -> int:
        """
        计算 Problem Solving 的符号调整

        规则：
        - 两个+ → +1
        - 一个+，没有- → +1
        - 一个+，一个- → 0（正负抵消）
        - 其他情况 → 0
        """
        c_sym = HayCalculator._extract_symbol(challenge)
        e_sym = HayCalculator._extract_symbol(environment)

        symbols = [c_sym, e_sym]
        plus_count = symbols.count('+')
        minus_count = symbols.count('-')

        # 有+号且没有被-号抵消
        if plus_count >= 1 and minus_count == 0:
            return 1
        # 其他情况（包括一个+一个-抵消，或全是-，或全是N）
        else:
            return 0

    @staticmethod
    def _calculate_acc_symbol_adjustment(freedom: str, magnitude: str, nature: str) -> int:
        """计算 Accountability 的符号调整"""
        f_sym = HayCalculator._extract_symbol(freedom)
        m_sym = HayCalculator._extract_symbol(magnitude)
        n_sym = HayCalculator._extract_symbol(nature)
        
        symbols = [f_sym, m_sym, n_sym]
        plus_count = symbols.count('+')
        minus_count = symbols.count('-')

        # 负号占主导 -> +0
        if minus_count >= 2: return 0
        if minus_count >= 1 and plus_count == 0: return 0

        # 正号占主导 -> +2
        if plus_count >= 2: return 2
        if plus_count >= 1 and minus_count == 0: return 2

        # 平衡 -> +1
        return 1

    def calculate(self, factors: HayFactors) -> HayResult:
        """
        执行完整的 Hay 评估计算
        """
        # 1. Know How Calculation
        kh_result = self._calculate_know_how(factors)

        # 2. Problem Solving Calculation
        ps_result = self._calculate_problem_solving(factors, kh_result.kh_level)

        # 3. Accountability Calculation
        acc_result = self._calculate_accountability(factors)

        # 4. Total Score & Grade
        total_score = kh_result.kh_score + ps_result.ps_score + acc_result.acc_score
        job_grade = get_job_grade(total_score)

        # 5. Calculate Job Profile (岗位特性)
        profile_data = calculate_job_profile(ps_result.ps_level, acc_result.acc_level)
        job_profile = JobProfile(**profile_data)

        # 6. Construct Result
        summary = HaySummary(
            kh_level=kh_result.kh_level,
            kh_score=kh_result.kh_score,
            ps_base_level=ps_result.ps_base_level,
            ps_percentage=ps_result.ps_percentage_str,
            ps_level=ps_result.ps_level,
            ps_score=ps_result.ps_score,
            acc_level=acc_result.acc_level,
            acc_score=acc_result.acc_score,
            total_score=total_score,
            job_grade=job_grade,
            job_profile=job_profile
        )

        return HayResult(
            input_factors=factors,
            know_how=kh_result,
            problem_solving=ps_result,
            accountability=acc_result,
            total_score=total_score,
            job_grade=job_grade,
            summary=summary
        )

    def _calculate_know_how(self, factors: HayFactors) -> KnowHowResult:
        p_score = PRACTICAL_KNOWLEDGE_SCORES[factors.practical_knowledge]
        m_score = MANAGERIAL_KNOWLEDGE_SCORES[factors.managerial_knowledge]
        c_score = COMMUNICATION_SCORES[factors.communication]

        base_level = p_score + m_score + c_score
        adj = self._calculate_kh_symbol_adjustment(factors.practical_knowledge, factors.managerial_knowledge)
        
        kh_level = base_level + adj
        kh_score = get_level_score(kh_level)  # 使用安全函数

        return KnowHowResult(
            practical_score=p_score,
            managerial_score=m_score,
            communication_score=c_score,
            base_level=base_level,
            symbol_adjustment=adj,
            kh_level=kh_level,
            kh_score=kh_score
        )

    def _calculate_problem_solving(self, factors: HayFactors, kh_level: int) -> ProblemSolvingResult:
        c_score = THINKING_CHALLENGE_SCORES[factors.thinking_challenge]
        e_score = THINKING_ENVIRONMENT_SCORES[factors.thinking_environment]

        base_level = c_score + e_score
        adj = self._calculate_ps_symbol_adjustment(factors.thinking_challenge, factors.thinking_environment)
        
        ps_base_level = base_level + adj

        # 安全获取PS百分比（边界处理）
        if ps_base_level < 1:
            ps_base_level = 1
        elif ps_base_level > 19:
            ps_base_level = 19
        ps_percentage = PS_PERCENTAGE_TABLE.get(ps_base_level, 0.87)  # 默认87%

        # 使用安全的矩阵查找函数
        ps_level = get_ps_level_from_matrix(ps_base_level, kh_level)
        ps_score = get_level_score(ps_level)  # 使用安全函数

        return ProblemSolvingResult(
            challenge_score=c_score,
            environment_score=e_score,
            base_level=base_level,
            symbol_adjustment=adj,
            ps_base_level=ps_base_level,
            ps_percentage=ps_percentage,
            ps_percentage_str=f"{int(ps_percentage * 100)}%",
            ps_level=ps_level,
            ps_score=ps_score
        )

    def _calculate_accountability(self, factors: HayFactors) -> AccountabilityResult:
        f_score = FREEDOM_TO_ACT_SCORES[factors.freedom_to_act]
        
        is_quantified = factors.magnitude != 'N'
        m_score = MAGNITUDE_SCORES_QUANTIFIED[factors.magnitude]

        if is_quantified:
            n_score = NATURE_OF_IMPACT_SCORES_QUANTIFIED[factors.nature_of_impact]
        else:
            n_score = NATURE_OF_IMPACT_SCORES_NON_QUANTIFIED[factors.nature_of_impact]

        base_level = f_score + m_score + n_score
        adj = self._calculate_acc_symbol_adjustment(factors.freedom_to_act, factors.magnitude, factors.nature_of_impact)
        
        acc_level = base_level + adj
        acc_score = get_level_score(acc_level)  # 使用安全函数

        return AccountabilityResult(
            freedom_score=f_score,
            magnitude_score=m_score,
            nature_score=n_score,
            base_level=base_level,
            symbol_adjustment=adj,
            acc_level=acc_level,
            acc_score=acc_score,
            is_quantified=is_quantified
        )

# Helper for backward compatibility or easy usage
def calculate_hay_evaluation(factors: dict) -> dict:
    """
    Wrapper to maintain backward compatibility but use the new robust engine.
    """
    # Validate input using Pydantic
    try:
        validated_factors = HayFactors(**factors)
    except Exception as e:
        # Re-raise as a clearer error or let it bubble up
        raise ValueError(f"Input validation failed: {e}")

    calculator = HayCalculator()
    result = calculator.calculate(validated_factors)
    
    # Convert back to dict to match old return signature if needed, 
    # or just return the model's dict representation
    return result.model_dump()
