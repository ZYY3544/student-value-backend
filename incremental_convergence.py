"""
===========================================
增量收敛引擎 (Incremental Convergence Engine)
===========================================
版本：v4.3（单阶段策略 + 完全规则驱动 + 直接保留规则表符号）

作用：从易于识别的因素开始，逐步收敛到符合职能常模的最优解

核心策略（8个步骤）：
1. PK+RSCP智能提取（根据量化模式优化）：
   - 可量化模式：LLM一次调用提取PK和RSCP（业务场景判断）
   - 不可量化模式：LLM只提取PK档位（其他因素规则推导）
2. 职能常模：查询目标Profile列表（如人力资源→['L','A1']）
3. KH收敛：基于PK直接筛选，从171个100%合法组合收敛到~24个
4. PS精确收敛：基于PS百分比反向映射，直接查表生成(TE,TC)档位（~48个组合，保留规则表符号）
5. Magnitude确定：规则驱动（可量化→1-5档，不可量化→N档）
6. ACC因素确定（完全基于规则推导）：
   - Freedom：从TE反推（基于约束链 PK≥TE≥Freedom，带符号）
   - Nature：可量化模式来自Step 1的RSCP，不可量化模式从ACC规则表筛选
7. 方案生成：生成完整方案（8因素，保留规则表符号，40-80个候选）
8. 验证并排序：约束链验证（PK≥TE≥Freedom）+ HAY计算 + 按Profile匹配度排序

关键优势：
- 收敛率：99.9999%+（从~45亿组合 → 30-80个有效方案）
- 性能提升：50倍+（KH缓存 + 渐进式验证 + 单阶段策略）
- LLM调用极少：可量化模式1次（PK+RSCP），不可量化模式1次（PK）【Step 6完全规则驱动】
- 约束链完整应用：PK ≥ TE ≥ Freedom（在PS收敛和ACC确定中自动应用）
- 完全规则驱动：PS精确收敛、TE/TC查表生成（保留符号）、Freedom从TE反推、Nature规则筛选
- 符号100%合法：直接保留规则表符号，无需人工构造和二次验证
"""

from typing import Dict, List, Tuple

from validation_rules import ValidationRules
from profile_norm_validator import profile_norm_validator
from calculator import HayCalculator
from logger import get_module_logger
from magnitude_mapper import map_amount_to_magnitude

logger = get_module_logger(__name__)


def determine_magnitude_from_amount(amount: float) -> str:
    """
    根据金额（万元）确定影响范围档位（带+/-后缀，使用对数均分法）

    Args:
        amount: 金额（万元）

    Returns:
        档位字符串（带+/-后缀）：'1-', '1', '1+', '2-', '2', '2+', ..., '9-', '9', '9+', 'N'

    Examples:
        - 3亿元(30000万) → '4-'（靠近下限）
        - 10亿元(100000万) → '4'（中间）
        - 25亿元(250000万) → '4+'（靠近上限）
    """
    return map_amount_to_magnitude(amount)


class IncrementalConvergence:
    """增量收敛引擎"""

    def __init__(self, validation_rules: ValidationRules, llm_service=None):
        """
        初始化增量收敛引擎

        Args:
            validation_rules: 验证规则实例
            llm_service: 可选的LLM服务实例（用于语义理解）
        """
        self.validation_rules = validation_rules
        self.calculator = HayCalculator()
        self.llm_service = llm_service

        # HAY计算缓存（优化性能）
        self.kh_ps_cache = {}  # 缓存KH和PS的计算结果
        self.cache_hits = 0    # 统计缓存命中次数
        self.cache_misses = 0  # 统计缓存未命中次数


    def converge_ps_combinations(
        self,
        kh_combinations: List[Dict]
    ) -> List[Dict]:
        """
        第二步：收敛PS合法组合（精确版：基于PS百分比直接查询）

        新策略：
        1. KH组合 → KH Score
        2. 查PS×KH规则 → 合法的PS百分比
        3. 从ps_percentage_mapping直接查询 → (TE, TC)组合
        4. 应用约束：TE ≤ PK
        5. 完成！

        优势：完全消除无效候选，只生成真正合法的组合

        Returns:
            [
                {
                    'thinking_challenge': '3',
                    'thinking_environment': 'C',
                    'kh_combo': {...},
                    'kh_score': 304,
                    'kh_level': 11,
                    'ps_level': 9
                },
                ...
            ]
        """
        logger.info(f"[精确PS收敛] 开始 - KH组合数: {len(kh_combinations)}")

        valid_ps_combinations = []

        # Phase 1: 预计算KH Score和KH Level（已有逻辑，保持不变）
        from models import HayFactors
        from data_tables import SCORE_LEVEL_TABLE, LEVEL_SCORE_TABLE, PS_MATRIX

        kh_score_cache = {}
        kh_level_cache = {}

        logger.info(f"[Phase 1] 预计算KH Score - {len(kh_combinations)} 个组合")
        for kh_combo in kh_combinations:
            try:
                temp_factors = HayFactors(
                    practical_knowledge=kh_combo['practical_knowledge'],
                    managerial_knowledge=kh_combo['managerial_knowledge'],
                    communication=kh_combo['communication'],
                    thinking_challenge='1', thinking_environment='A',
                    freedom_to_act='A', magnitude='N', nature_of_impact='I'
                )
                kh_result = self.calculator.calculate(temp_factors)
                kh_score = kh_result.know_how.kh_score

                kh_key = (kh_combo['practical_knowledge'],
                         kh_combo['managerial_knowledge'],
                         kh_combo['communication'])
                kh_score_cache[kh_key] = kh_score

                # 查找KH Level (SCORE_LEVEL_TABLE 是 {score: level} 格式)
                kh_level = None
                for score, level in SCORE_LEVEL_TABLE.items():
                    if score == kh_score:
                        kh_level = level
                        break

                if kh_level is None:
                    closest_level = min(LEVEL_SCORE_TABLE.keys(),
                                      key=lambda x: abs(LEVEL_SCORE_TABLE[x] - kh_score))
                    kh_level = closest_level

                kh_level_cache[kh_key] = kh_level

            except Exception as e:
                logger.error(f"计算KH Score失败: {e}")
                continue

        logger.info(f"[Phase 1] KH Score缓存完成 - {len(kh_score_cache)} 个")

        # Phase 2: 精确反推(TE, TC)组合（保留规则表中的符号）
        total_valid_combos = 0
        seen_combinations = set()  # 用于去重

        for kh_combo in kh_combinations:
            kh_key = (kh_combo['practical_knowledge'],
                     kh_combo['managerial_knowledge'],
                     kh_combo['communication'])

            if kh_key not in kh_level_cache:
                continue

            kh_score = kh_score_cache[kh_key]
            kh_level = kh_level_cache[kh_key]
            pk = kh_combo['practical_knowledge']

            # Step 2.1: 从PS×KH规则中找出合法的PS百分比
            legal_ps_percentages = set()
            for (ps_pct_str, kh_score_in_rule), (result, prob) in self.validation_rules.ps_kh_rules.items():
                if prob == '100%' and kh_score_in_rule == kh_score:
                    legal_ps_percentages.add(ps_pct_str)

            if not legal_ps_percentages:
                continue

            # Step 2.2: 直接从ps_percentage_mapping查询对应的(TE, TC)组合
            for ps_percentage in legal_ps_percentages:
                if ps_percentage not in self.validation_rules.ps_percentage_mapping:
                    continue

                te_tc_combinations = self.validation_rules.ps_percentage_mapping[ps_percentage]

                for te, tc in te_tc_combinations:
                    # Step 2.3: 应用约束：TE 必须是 PK 紧挨着的下一格（序列相邻）
                    # 例如：PK='E' → TE 只能是 'E-'；PK='E-' → TE 只能是 'D+'
                    if not self._is_adjacent_below(te, pk):
                        continue

                    # Step 2.4: 去重检查（保留符号）
                    combo_key = (kh_key, te, tc)
                    if combo_key in seen_combinations:
                        continue
                    seen_combinations.add(combo_key)

                    # Step 2.5: 计算PS Level（含符号调整）
                    from data_tables import THINKING_CHALLENGE_SCORES, THINKING_ENVIRONMENT_SCORES
                    from utils import calculate_ps_symbol_adjustment

                    # 提取基础档位
                    tc_base = self._strip_symbol(tc)
                    te_base = self._strip_symbol(te)

                    c_score = THINKING_CHALLENGE_SCORES.get(tc_base, 0)
                    e_score = THINKING_ENVIRONMENT_SCORES.get(te_base, 0)

                    # PS base level
                    ps_base_level = c_score + e_score

                    # 符号调整（统一使用utils.py中的函数，与calculator.py保持一致）
                    ps_adjustment = calculate_ps_symbol_adjustment(tc, te)

                    ps_level_with_adjustment = ps_base_level + ps_adjustment

                    # 从PS_MATRIX查表
                    try:
                        ps_level = PS_MATRIX[ps_level_with_adjustment][kh_level]
                    except KeyError:
                        logger.warning(f"PS_MATRIX查表失败: ps_level={ps_level_with_adjustment}, kh_level={kh_level}")
                        continue

                    valid_ps_combinations.append({
                        'thinking_challenge': tc,  # 保留符号，如'3+'
                        'thinking_environment': te,  # 保留符号，如'E-'
                        'kh_combo': kh_combo,
                        'kh_score': kh_score,
                        'kh_level': kh_level,
                        'ps_level': ps_level
                    })
                    total_valid_combos += 1

        logger.info(f"[Phase 2] 精确PS收敛完成 - 生成 {total_valid_combos} 个有效PS+KH组合（保留规则表原始符号）")
        logger.info(f"  优化策略：直接使用规则表符号，100%合法组合保证")

        return valid_ps_combinations

    @staticmethod
    def _strip_symbol(level: str) -> str:
        """去除档位的符号，返回基础档位"""
        return level.rstrip('+-')

    @staticmethod
    def _is_adjacent_below(lower: str, upper: str) -> bool:
        """检查 lower 是否是 upper 在有序序列中紧挨着的下一格（严格小于且相邻）

        有序序列：A-, A, A+, B-, B, B+, ..., H-, H, H+
        例如：
        - _is_adjacent_below('E-', 'E')  → True   (E- 紧挨在 E 下方)
        - _is_adjacent_below('D+', 'E-') → True   (D+ 紧挨在 E- 下方)
        - _is_adjacent_below('D',  'E')  → False  (中间还有 D+, E-)
        """
        all_levels = []
        for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            all_levels.extend([letter + '-', letter, letter + '+'])

        if lower not in all_levels or upper not in all_levels:
            return False

        return all_levels.index(upper) - all_levels.index(lower) == 1

    @staticmethod
    def _level_to_numeric(level: str) -> float:
        """
        将档位（含符号）转换为数值，用于精确距离计算

        支持两种档位类型：
        1. 字母档位（A-H）：用于PK, TE, Freedom等
        2. 数字档位（1-5）：用于TC (Thinking Challenge)

        符号调整使用 ±0.3（而非 ±0.5），确保相邻档位有明确的大小关系：
        - E- (4.7) > D+ (4.3)，而非 E- = D+ = 4.5
        - 这样可以保持严格小于约束，同时不丢失合法组合

        例如：
        - 'E' → 5.0
        - 'E+' → 5.3
        - 'E-' → 4.7
        - 'D+' → 4.3
        - 'D' → 4.0
        - '3' → 3.0 (TC档位)
        - '3+' → 3.3 (TC档位)

        Args:
            level: 档位字符串，如'E', 'E+', 'D-', '3', '3+'

        Returns:
            数值表示（基础档位值 + 符号调整），无效档位返回-1.0
        """
        # 字母档位顺序（用于PK, TE, Freedom等）
        letter_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        # 数字档位顺序（用于TC）
        digit_order = ['1', '2', '3', '4', '5']

        base = level.rstrip('+-')

        # 判断是字母档位还是数字档位
        if base in letter_order:
            base_value = float(letter_order.index(base) + 1)  # A=1, B=2, ..., H=8
        elif base in digit_order:
            base_value = float(base)  # '1'→1.0, '2'→2.0, ..., '5'→5.0
        else:
            return -1.0

        # 符号调整（使用 ±0.3，确保 E- > D+）
        if level.endswith('+'):
            return base_value + 0.3
        elif level.endswith('-'):
            return base_value - 0.3
        else:
            return base_value

    @staticmethod
    def _select_pk_by_profile_combination(pk_range: List[str], target_profiles: List[str]) -> str:
        """
        根据职能常模组合智能选择单一PK档位（极致收敛策略）

        用户定义的策略（以['E', 'F']为例）：
        - ['P2', 'P3'] → 选 'F'（高档，提高PS）
        - ['P1', 'L'] 或 ['L', 'A1'] 或 ['A1', 'A2'] → 选 'E+'（中档）
        - ['A2', 'A3'] → 选 'E'（低档，降低PS）

        核心逻辑：
        - P型常模需要高PS → 选高PK
        - A型常模需要低PS（让ACC>PS）→ 选低PK

        Args:
            pk_range: LLM提取的PK范围，如['E', 'F']
            target_profiles: 职能常模，如['A1', 'A2']

        Returns:
            单一PK档位（可能带符号），如'E+'
        """
        # 如果只有一个PK，直接返回
        if len(pk_range) == 1:
            return pk_range[0]

        # 标准化常模列表（排序后作为key）
        sorted_profiles = sorted(target_profiles)
        profile_key = ';'.join(sorted_profiles)

        # 排序PK范围（从低到高）
        pk_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        sorted_pks = sorted(pk_range, key=lambda x: pk_order.index(x) if x in pk_order else 999)

        # 精确匹配策略表（用户定义）
        pk_strategy_map = {
            'P2;P3': ('high', None),        # 选高档（不带符号）
            'L;P1': ('mid_high', '+'),      # 选低档+符号
            'A1;L': ('mid_high', '+'),      # 选低档+符号
            'A1;A2': ('mid_high', '+'),     # 选低档+符号
            'A2;A3': ('low', None),         # 选低档（不带符号）
        }

        # 查找匹配的策略
        strategy_tuple = pk_strategy_map.get(profile_key)

        if strategy_tuple:
            strategy, symbol = strategy_tuple
            if strategy == 'high':
                # 选高档：F
                selected = sorted_pks[-1]
                logger.info(f"    匹配策略: {profile_key} → 高档PK（P型常模，提高PS）")
            elif strategy == 'mid_high':
                # 选低档+符号：E+
                selected = sorted_pks[0] + (symbol if symbol else '')
                logger.info(f"    匹配策略: {profile_key} → 中档PK（混合常模，平衡PS和ACC）")
            elif strategy == 'low':
                # 选低档：E
                selected = sorted_pks[0]
                logger.info(f"    匹配策略: {profile_key} → 低档PK（A型常模，降低PS）")
            else:
                selected = sorted_pks[0]
        else:
            # 未匹配到精确策略，使用通用规则
            logger.info(f"    未找到精确匹配策略'{profile_key}'，使用通用规则")

            # 判断常模类型
            has_a = any(p.startswith('A') for p in target_profiles)
            has_p = any(p.startswith('P') for p in target_profiles)
            has_l = 'L' in target_profiles

            if has_p and not has_a and not has_l:
                # 纯P型 → 选高档（提高PS）
                selected = sorted_pks[-1]
                logger.info(f"    通用规则: 纯P型常模 → 高档PK")
            elif has_a and not has_p and not has_l:
                # 纯A型 → 选低档（降低PS）
                selected = sorted_pks[0]
                logger.info(f"    通用规则: 纯A型常模 → 低档PK")
            else:
                # 混合型或L型 → 选中档
                selected = sorted_pks[0] + '+'
                logger.info(f"    通用规则: 混合/L型常模 → 中档PK")

        return selected

    def _infer_freedom_from_te(self, te: str) -> List[str]:
        """从TE反推Freedom候选范围（严格小于，紧邻一格）

        基于约束链：PK > TE > Freedom（严格大于）
        策略：在完整档位序列中找到 TE 的紧邻下一格

        完整档位序列（从低到高）：
        A-, A, A+, B-, B, B+, C-, C, C+, D-, D, D+, E-, E, E+, F-, F, F+, G-, G, G+

        例如：
        - TE='E'  → Freedom='E-'（同字母内下一格，距离0.3）
        - TE='E-' → Freedom='D+'（跨字母边界下一格，距离0.4）
        - TE='D+' → Freedom='D'（同字母内下一格，距离0.3）

        Args:
            te: 思考环境档位（含符号，如'E-', 'D'等）

        Returns:
            Freedom候选列表（仅1个，紧邻下一格）
        """
        # 完整档位序列（从低到高排列）
        all_levels = []
        for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            all_levels.extend([letter + '-', letter, letter + '+'])

        if te not in all_levels:
            logger.warning(f"未知TE档位'{te}'，使用默认Freedom: C-")
            return ['C-']

        idx = all_levels.index(te)

        if idx == 0:
            logger.info(f"特殊情况：TE='{te}'已是最低档，Freedom只能取A-")
            return ['A-']

        candidate = all_levels[idx - 1]
        logger.info(f"  - Freedom反推: TE='{te}' → Freedom='{candidate}'（紧邻下一格）")
        return [candidate]

    def _apply_grade_education_floor(
        self,
        valid_solutions: List[Tuple[Dict, float, str, int]],
        best_solution: Dict,
        best_score: float,
        education: str
    ) -> Tuple[Dict, float]:
        """根据学历对最终职级应用下限约束（仅在读状态有效）

        规则：
        - 本科在读: grade >= 9
        - 研究生在读: grade >= 10
        - 其他（毕业 / 博士 / 未知）: 不干预，走原收敛策略
        """
        education_grade_floor = {
            '本科在读': 9,
            '研究生在读': 10,
        }
        floor = education_grade_floor.get(education)
        if not floor:
            return best_solution, best_score

        # 找到当前最优解对应的职级
        best_grade = None
        for sol, score, profile, grade in valid_solutions:
            if sol is best_solution:
                best_grade = grade
                break

        if best_grade is None or best_grade >= floor:
            return best_solution, best_score

        # 当前职级低于下限，从 grade >= floor 的方案中重新选择
        floor_solutions = [
            (sol, score, profile, grade)
            for sol, score, profile, grade in valid_solutions
            if grade >= floor
        ]

        if not floor_solutions:
            logger.warning(f"⚠️ 学历职级兜底: 无grade>={floor}的方案，保持原选择（grade={best_grade}）")
            return best_solution, best_score

        logger.info(f"  - 学历职级兜底: grade {best_grade} → 最低{floor}（学历: {education}）")
        return self._select_solution_by_grade_strategy(floor_solutions)

    def _apply_pk_education_floor(self, pk: str, education: str) -> str:
        """根据学历对 PK 档位应用下限约束

        学历 → PK 最低档位：
        - 本科在读: C-
        - 本科毕业: C-
        - 研究生在读: C
        - 研究生毕业: D
        - 博士在读: D+
        - 博士毕业: E-
        - 未知: 不干预
        """
        education_pk_floor = {
            '本科在读': 'C-',
            '本科毕业': 'C-',
            '研究生在读': 'C',
            '研究生毕业': 'D',
            '博士在读': 'D+',
            '博士毕业': 'E-',
        }
        floor = education_pk_floor.get(education)
        if not floor:
            return pk
        pk_value = self._level_to_numeric(pk)
        floor_value = self._level_to_numeric(floor)
        if pk_value < floor_value:
            logger.info(f"  - PK学历兜底: {pk} → {floor}（学历: {education}）")
            return floor
        return pk

    # =====================================================================
    # 注意：反推ACC范围的方法已废弃（2025-12-28）
    # 原因：ACC应该从JD内容分析，而不是从职能常模反推
    # 当前策略：使用LLM直接提取Freedom和Nature范围，Magnitude由规则确定
    # 历史实现参考：git history 或 INCREMENTAL_CONVERGENCE_DESIGN.md
    # =====================================================================

    def generate_base_solutions_phase1(
        self,
        ps_kh_combinations: List[Dict],
        acc_hints: Dict,
        magnitude: str
    ) -> List[Dict]:
        """
        生成完整方案（8因素，保留规则表符号）

        策略：
        - 输入：PS+KH组合（PK, MK, Comm, TE, TC 带规则表符号）
        - 输出：完整方案（8因素全部带规则表符号）
        - 优势：直接使用规则表符号，100%合法，无需二次微调

        Args:
            ps_kh_combinations: PS+KH组合列表（带规则表符号）
            acc_hints: ACC因素候选范围（带符号）
                {
                    'freedom_to_act': ['D', 'D-', 'C+', 'C'],
                    'nature_of_impact': ['III', 'IV']
                }
            magnitude: 影响范围（规则确定：'N' 或 '1'-'5'）

        Returns:
            完整方案列表（8因素，全部带规则表符号）
            [
                {
                    'practical_knowledge': 'E',      # 或 'E+', 'E-'
                    'managerial_knowledge': 'II',    # 或 'II+', 'II-'
                    'communication': '2',
                    'thinking_challenge': '4',       # 或 '4+', '4-'
                    'thinking_environment': 'E',     # 或 'E+', 'E-'
                    'freedom_to_act': 'D',          # 或 'D+', 'D-'
                    'magnitude': 'N',
                    'nature_of_impact': 'III'       # 或 'III+', 'III-'
                },
                ...
            ]
        """
        logger.info(f"[方案生成] 开始生成完整方案（保留规则表符号） - PS+KH组合数: {len(ps_kh_combinations)}")
        logger.info(f"  - Magnitude（规则确定）: {magnitude}")
        logger.info(f"  - Freedom范围（从TE反推）: {acc_hints.get('freedom_to_act', [])}")
        logger.info(f"  - Nature范围（规则推导/RSCP）: {acc_hints.get('nature_of_impact', [])}")

        base_solutions = []

        # 从acc_hints获取Freedom和Nature候选范围（现在已经是带符号档位）
        freedom_candidates = acc_hints.get('freedom_to_act', ['B', 'C'])
        nature_candidates = acc_hints.get('nature_of_impact', ['II', 'III'])

        logger.info(f"  - Freedom候选范围: {freedom_candidates}")
        logger.info(f"  - Nature候选范围: {nature_candidates}")

        # 预先过滤有效的(freedom, nature)组合（用基础档位验证）
        valid_freedom_nature_pairs = []
        invalid_count = 0

        for freedom in freedom_candidates:
            for nature in nature_candidates:
                # 提前验证ACC组合（magnitude='N'时需要验证，使用基础档位）
                if magnitude == 'N':
                    freedom_base = self._strip_symbol(freedom)
                    nature_base = self._strip_symbol(nature)
                    is_valid, _, prob = self.validation_rules.validate_acc(
                        freedom_base, magnitude, nature_base
                    )
                    if not is_valid or prob != '100%':
                        invalid_count += 1
                        continue

                valid_freedom_nature_pairs.append((freedom, nature))

        logger.info(f"  - ACC组合预过滤: {len(freedom_candidates) * len(nature_candidates)} → {len(valid_freedom_nature_pairs)} 个有效组合")

        # 生成完整方案（应用约束链：PK ≥ TE ≥ Freedom，保留规则表符号）
        constraint_violations = 0
        for ps_kh_combo in ps_kh_combinations:
            te = ps_kh_combo['thinking_environment']
            pk = ps_kh_combo['kh_combo']['practical_knowledge']

            for freedom, nature in valid_freedom_nature_pairs:
                # ❗ 关键约束：Freedom 必须是 TE 紧挨着的下一格（序列相邻）
                # TE < PK 已在PS收敛时验证，此处验证 Freedom < TE
                if not self._is_adjacent_below(freedom, te):
                    constraint_violations += 1
                    continue  # 跳过违反约束的组合

                # 构建完整方案（8因素，保留规则表原始符号）
                base_solution = {
                    **ps_kh_combo['kh_combo'],  # PK, MK, Comm（带符号）
                    'thinking_challenge': ps_kh_combo['thinking_challenge'],  # 带符号
                    'thinking_environment': te,  # 带符号
                    'freedom_to_act': freedom,  # 带符号
                    'magnitude': magnitude,
                    'nature_of_impact': nature  # 带符号
                }

                base_solutions.append(base_solution)

        if constraint_violations > 0:
            logger.info(f"  - 约束链过滤: 跳过 {constraint_violations} 个违反 Freedom<TE 约束的组合（严格小于）")

        logger.info(f"[方案生成] 完成 - 共生成 {len(base_solutions)} 个完整方案（8因素，保留规则表符号）")
        logger.info(f"  计算: {len(ps_kh_combinations)} PS+KH组合 × {len(valid_freedom_nature_pairs)} ACC组合 = {len(base_solutions)} 完整方案")

        return base_solutions

    def validate_and_rank_solutions(
        self,
        candidates: List[Dict],
        function: str,
        target_profiles: List[str]
    ) -> List[Tuple[Dict, float]]:
        """
        第五步：验证并排序候选解（简化版：移除冗余验证）

        优化说明：
        - 增量收敛引擎（Step 3-7）已经保证了KH、PS、ACC、PS×KH规则的合法性
        - 这里只需保留约束链验证（PK > TE > Freedom，严格大于）作为最后的安全网
        - 然后计算HAY分数和匹配度进行排序

        流程：
        1. 约束链验证（PK > TE > Freedom，严格大于）- 防御性检查
        2. HAY计算（获取Profile和职级）
        3. 计算匹配度并排序

        Returns:
            [(solution, match_score, profile, job_grade), ...]  按match_score降序排列
        """
        logger.info(f"开始验证和排序 - 候选数: {len(candidates)}, 目标Profile: {target_profiles}")

        validated_solutions = []
        constraint_chain_violations = 0

        for candidate in candidates:
            # 约束链验证（PK > TE > Freedom，严格大于）
            # 这是增量收敛中最容易漏掉的约束，作为最后的安全网
            pk = candidate['practical_knowledge']
            te = candidate['thinking_environment']
            freedom = candidate['freedom_to_act']

            # 检查 TE 是 PK 紧挨着的下一格
            if not self._is_adjacent_below(te, pk):
                constraint_chain_violations += 1
                logger.debug(f"约束链违反: TE({te}) 不是 PK({pk}) 的紧邻下一格")
                continue

            # 检查 Freedom 是 TE 紧挨着的下一格
            if not self._is_adjacent_below(freedom, te):
                constraint_chain_violations += 1
                logger.debug(f"约束链违反: Freedom({freedom}) 不是 TE({te}) 的紧邻下一格")
                continue

            # 计算HAY分数（带缓存优化）
            try:
                hay_result = self._calculate_full_score(candidate)

                actual_profile = hay_result['summary']['job_profile']['profile_type']
                job_grade = hay_result['job_grade']

                # 计算匹配度
                match_score = self._calculate_profile_match_score(
                    actual_profile, target_profiles
                )

                validated_solutions.append((candidate, match_score, actual_profile, job_grade))

            except Exception as e:
                logger.error(f"计算评分失败: {e}")
                continue

        # 按匹配度排序（降序）
        validated_solutions.sort(key=lambda x: x[1], reverse=True)

        # 日志输出
        logger.info(f"验证完成 - {len(validated_solutions)}/{len(candidates)} 个候选通过验证")
        if constraint_chain_violations > 0:
            logger.info(f"  - 约束链过滤: {constraint_chain_violations} 个（增量收敛遗漏）")

        # 计算过滤率
        filter_rate = (len(candidates) - len(validated_solutions)) / len(candidates) * 100 if len(candidates) > 0 else 0
        if filter_rate > 0:
            logger.info(f"  - 过滤率: {filter_rate:.1f}%")

        logger.info(f"说明: Layer 1-4验证已移除（增量收敛已保证合法性），仅保留约束链安全网")

        return [(sol, score, profile, grade) for sol, score, profile, grade in validated_solutions]


    def _calculate_full_score(self, solution: Dict) -> Dict:
        """
        计算完整的HAY评分（带缓存优化）

        缓存策略：
        - 缓存键：全部8个因素（PK, MK, Comm, TC, TE, Freedom, Magnitude, Nature）
        - 缓存值：(KH分数, PS百分比, KH Level, PS Level, ACC Level, Profile类型)
        - 命中缓存：跳过昂贵的calculator.calculate()调用
        - 重要：Profile依赖于Level Gap = PS_Level - ACC_Level，而ACC_Level依赖于Freedom/Magnitude/Nature

        Returns:
            包含完整HAY计算结果的字典，格式：
            {
                'summary': {...},
                'kh_score': int,
                'ps_percentage': float,
                'ps_percentage_str': str,
                'job_grade': int,
                '_from_cache': bool
            }
        """
        from models import HayFactors

        # 构建缓存键（全部8个因素，因为Profile依赖ACC Level）
        cache_key = (
            solution['practical_knowledge'],
            solution['managerial_knowledge'],
            solution['communication'],
            solution['thinking_challenge'],
            solution['thinking_environment'],
            solution['freedom_to_act'],
            solution['magnitude'],
            solution['nature_of_impact']
        )

        # 检查缓存
        if cache_key in self.kh_ps_cache:
            # ✅ 缓存命中！
            self.cache_hits += 1
            cached = self.kh_ps_cache[cache_key]

            logger.debug(f"[缓存命中 {self.cache_hits}] KH={cached['kh_score']}, PS={cached['ps_percentage_str']}, Profile={cached['profile_type']}")

            # 从缓存返回结果
            return {
                'summary': {
                    'job_profile': {
                        'profile_type': cached['profile_type'],
                        'description': cached['description']
                    }
                },
                'kh_score': cached['kh_score'],
                'ps_percentage': cached['ps_percentage'],
                'ps_percentage_str': cached['ps_percentage_str'],
                'job_grade': cached['job_grade'],
                '_from_cache': True
            }

        # ❌ 缓存未命中，执行完整计算
        self.cache_misses += 1

        factors = HayFactors(**solution)
        result = self.calculator.calculate(factors)

        # 提取关键信息
        kh_score = result.know_how.kh_score
        ps_percentage = result.problem_solving.ps_percentage
        ps_percentage_str = f"{int(ps_percentage * 100)}%"
        profile_type = result.summary.job_profile.profile_type if result.summary.job_profile else 'Unknown'
        description = result.summary.job_profile.description if result.summary.job_profile else '未知'
        job_grade = result.summary.job_grade

        # 存入缓存
        self.kh_ps_cache[cache_key] = {
            'kh_score': kh_score,
            'ps_percentage': ps_percentage,
            'ps_percentage_str': ps_percentage_str,
            'profile_type': profile_type,
            'description': description,
            'job_grade': job_grade
        }

        logger.debug(f"[缓存存储 {self.cache_misses}] {cache_key[:3]}... → KH={kh_score}, PS={ps_percentage_str}, Profile={profile_type}")

        # 返回完整结果（兼容旧接口 + 新增字段）
        return {
            'summary': {
                'job_profile': {
                    'profile_type': profile_type,
                    'description': description
                }
            },
            'kh_score': kh_score,
            'ps_percentage': ps_percentage,
            'ps_percentage_str': ps_percentage_str,
            'job_grade': job_grade,
            '_from_cache': False
        }


    def _validate_with_relaxed_constraints(
        self,
        candidates: List[Dict],
        function: str,
        target_profiles: List[str]
    ) -> List[Tuple[Dict, float, str, int]]:
        """放宽约束链验证: PK >= TE >= Freedom（允许相等）"""
        validated = []
        pk_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        te_order = ['A', 'B', 'C', 'D', 'E', 'F']
        freedom_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        for candidate in candidates:
            pk = candidate['practical_knowledge'].rstrip('+-')
            te = candidate['thinking_environment'].rstrip('+-')
            freedom = candidate['freedom_to_act'].rstrip('+-')

            # 放宽: 允许相等 (>=)
            pk_idx = pk_order.index(pk) if pk in pk_order else -1
            te_idx = te_order.index(te) if te in te_order else -1
            freedom_idx = freedom_order.index(freedom) if freedom in freedom_order else -1

            if pk_idx < 0 or te_idx < 0 or freedom_idx < 0:
                continue
            # PK档位 >= TE档位 >= Freedom档位 (字母序越小档位越高)
            if te_idx < pk_idx or freedom_idx < te_idx:
                continue

            try:
                hay_result = self._calculate_full_score(candidate)
                actual_profile = hay_result['summary']['job_profile']['profile_type']
                job_grade = hay_result['job_grade']
                match_score = self._calculate_profile_match_score(actual_profile, target_profiles)
                validated.append((candidate, match_score, actual_profile, job_grade))
            except Exception:
                continue

        validated.sort(key=lambda x: x[1], reverse=True)
        return validated

    def _validate_no_constraints(self, candidates: List[Dict]) -> List[Tuple[Dict, float, str, int]]:
        """无约束兜底: 跳过约束链，直接计算HAY评分"""
        validated = []
        for candidate in candidates[:50]:  # 最多取50个避免太慢
            try:
                hay_result = self._calculate_full_score(candidate)
                actual_profile = hay_result['summary']['job_profile']['profile_type']
                job_grade = hay_result['job_grade']
                validated.append((candidate, 0.0, actual_profile, job_grade))
            except Exception:
                continue
        return validated

    @staticmethod
    def _expand_target_profiles(target_profiles: List[str], profile_order: List[str], expand_steps: int = 2) -> List[str]:
        """逐步扩展常模范围，例如 [L, A1] 扩展2步 → [P2, P1, L, A1, A2, A3]"""
        indices = []
        for p in target_profiles:
            if p in profile_order:
                indices.append(profile_order.index(p))
        if not indices:
            return target_profiles

        min_idx = max(0, min(indices) - expand_steps)
        max_idx = min(len(profile_order) - 1, max(indices) + expand_steps)
        return profile_order[min_idx:max_idx + 1]

    def _calculate_profile_match_score(
        self,
        actual_profile: str,
        target_profiles: List[str]
    ) -> float:
        """
        计算Profile匹配度分数

        策略：
        1. 完全匹配 → 100分
        2. 同类型不同级别 → 90 - |级别差|*10
        3. 跨类型 → 50 - |类型差|*10

        Returns:
            匹配度分数 (0-100)
        """
        if actual_profile in target_profiles:
            return 100.0  # 完全匹配

        # 定义顺序
        profile_order = ['P4', 'P3', 'P2', 'P1', 'L', 'A1', 'A2', 'A3', 'A4']

        try:
            actual_idx = profile_order.index(actual_profile)
        except ValueError:
            return 0.0

        # 计算到所有目标Profile的最小距离
        min_distance = float('inf')
        for target in target_profiles:
            try:
                target_idx = profile_order.index(target)
                distance = abs(actual_idx - target_idx)

                # 判断是否同类型
                actual_type = actual_profile[0] if actual_profile != 'L' else 'L'
                target_type = target[0] if target != 'L' else 'L'

                if actual_type == target_type:
                    # 同类型，距离*5
                    score = 90 - distance * 5
                else:
                    # 跨类型，距离*10
                    score = 50 - distance * 10

                if score > 100 - min_distance:
                    min_distance = 100 - score

            except ValueError:
                continue

        return max(0, 100 - min_distance)

    def _select_solution_by_grade_strategy(
        self,
        valid_solutions: List[Tuple[Dict, float, str, int]]
    ) -> Tuple[Dict, float]:
        """
        两阶段筛选策略：先按Profile匹配度，再按职级选择

        阶段1：筛选出匹配度最高的所有方案
        阶段2：在这些方案中按职级策略选择

        职级选择策略：
        - 1个职级：唯一选择
        - 2个职级：选最低（index=1）
        - 3个职级：选最低（index=2）
        - 4个职级：选倒数第二个（index=2）
        - 其他：选最低

        Args:
            valid_solutions: [(solution, match_score, profile, job_grade), ...]
                            已按match_score降序排列

        Returns:
            (selected_solution, selected_score)
        """
        if not valid_solutions:
            raise ValueError("无有效方案可选")

        # ========== 阶段1：筛选出匹配度最高的所有方案 ==========
        best_match_score = valid_solutions[0][1]  # 第一个方案的匹配度（最高）

        top_matched_solutions = [
            (sol, score, profile, grade)
            for sol, score, profile, grade in valid_solutions
            if score == best_match_score
        ]

        logger.info(f"[两阶段选择] 阶段1：筛选出匹配度最高的方案")
        logger.info(f"  - 最高匹配度：{best_match_score:.1f}")
        logger.info(f"  - 该匹配度下有 {len(top_matched_solutions)} 个方案")

        # 如果只有1个最优匹配，直接返回
        if len(top_matched_solutions) == 1:
            logger.info(f"  - 唯一最优方案，直接选择")
            sol, score, profile, grade = top_matched_solutions[0]
            logger.info(f"  - 选中方案：Profile={profile}, 职级={grade}")
            return sol, score

        # ========== 阶段2：在最高匹配度的方案中按职级选择 ==========
        logger.info(f"[两阶段选择] 阶段2：在 {len(top_matched_solutions)} 个方案中按职级策略选择")

        # 按职级从高到低排序
        top_matched_solutions.sort(key=lambda x: x[3], reverse=True)

        # 收集唯一的职级
        unique_grades = []
        grade_to_solutions = {}  # {grade: [(sol, score, profile, grade), ...]}

        for sol, score, profile, grade in top_matched_solutions:
            if grade not in grade_to_solutions:
                unique_grades.append(grade)
                grade_to_solutions[grade] = []
            grade_to_solutions[grade].append((sol, score, profile, grade))

        grade_count = len(unique_grades)

        logger.info(f"  - 涉及 {grade_count} 个不同职级: {unique_grades}")

        # 应用职级选择策略（unique_grades 按降序排列：index 0=最高，index -1=最低）
        # - 1个职级：唯一选择
        # - 2个职级：选最低（index=1）
        # - 3个职级：选最低（index=2）
        # - 4个职级：选倒数第二个（index=2）
        # - 其他：选最低
        if grade_count == 4:
            selected_index = grade_count - 2  # 倒数第二个
        else:
            selected_index = grade_count - 1  # 最低职级
        logger.info(f"  - 策略：{grade_count}个职级 → 选第{selected_index+1}个（{'倒数第二' if grade_count == 4 else '最低'}职级）")

        selected_grade = unique_grades[selected_index]
        logger.info(f"  - 选中职级：{selected_grade}")

        # 在选中的职级中，选择第一个（因为已经按匹配度排序，同职级内任意一个都可以）
        candidates_in_grade = grade_to_solutions[selected_grade]
        logger.info(f"  - 该职级下有 {len(candidates_in_grade)} 个方案")

        # 选择该职级下的第一个方案
        sol, score, profile, grade = candidates_in_grade[0]

        logger.info(f"  ✓ 最终选择：Profile={profile}, 职级={grade}, 匹配度={score:.1f}")

        return sol, score

    def find_optimal_solution(
        self,
        eval_text: str,
        title: str,
        function: str,
        revenue_contribution: Dict = None,
        assessment_type: str = 'CV'
    ) -> Dict:
        """
        主函数：寻找最优解（仅支持 CV 模式）

        Args:
            eval_text: 简历内容
            title: 岗位名称
            function: 职能类型（用于查找常模）
            revenue_contribution: 营收贡献信息（可选）
            assessment_type: 评估类型（仅支持 'CV'）

        Returns:
            {
                'best_solution': {...},  # 最优八因素方案
                'match_score': 95.0,
                'all_valid_solutions': [...],  # 所有合法方案（按匹配度排序）
                'convergence_stats': {
                    'kh_combinations': 50,
                    'ps_kh_combinations': 200,
                    'candidates': 5000,
                    'valid_solutions': 120
                }
            }
        """
        import time as _time
        _ct = {}  # convergence timing
        _ct['start'] = _time.time()

        logger.info(f"=== 开始增量收敛（优化版：PK锚定策略）===")
        logger.info(f"岗位: {title}, 职能: {function}")

        # 检查LLM服务
        if not self.llm_service:
            error_msg = "系统需要LLM服务才能运行。请检查 GLM API 配置。"
            logger.error(f"[致命错误] {error_msg}")
            raise ValueError(error_msg)

        # Step 1: 调用LLM提取PK档位（带符号的单一档位）
        _ct['llm_start'] = _time.time()
        max_retries = 3
        last_exception = None
        pk_range = None
        logger.info("[Step 1] 提取PK档位（带符号，Nature使用默认值N）")

        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    logger.warning(f"[LLM] 第{attempt}次重试...")
                    import time
                    time.sleep(2 ** (attempt - 1))

                pk_result = self.llm_service.extract_pk_range(
                    eval_text=eval_text,
                    title=title,
                    function=function,
                    assessment_type=assessment_type
                )
                pk_range = pk_result['practical_knowledge']

                # 应用学历兜底约束
                education = pk_result.get('education', '未知')
                logger.info(f"  - 学历: {education}")
                pk_range = self._apply_pk_education_floor(pk_range, education)

                logger.info(f"✓ LLM提取成功 (第{attempt}次尝试)")
                logger.info(f"  - 专业知识档位: {pk_range}（带符号单一档位）")
                break

            except Exception as e:
                last_exception = e
                logger.warning(f"✗ LLM提取失败 (第{attempt}/{max_retries}次): {e}")

                if attempt == max_retries:
                    error_msg = f"LLM服务在{max_retries}次尝试后仍然失败: {last_exception}"
                    logger.error(f"[致命错误] {error_msg}")
                    raise RuntimeError(error_msg) from last_exception

        _ct['llm_end'] = _time.time()

        # Step 2: 获取职能常模
        target_profiles = profile_norm_validator.get_norm_profiles(function)
        if not target_profiles:
            logger.warning(f"职能'{function}'不在常模表中")
            target_profiles = ['L']  # 默认平衡型

        logger.info(f"[Step 2] 职能常模: {target_profiles}")

        # Step 2.5: 直接使用LLM返回的带符号PK档位
        # LLM已通过两步判断法给出带符号的单一档位（如 E、E+、E-）
        # 不再使用硬编码策略表 _select_pk_by_profile_combination
        logger.info(f"[Step 2.5] 使用LLM判断的PK档位")
        logger.info(f"  - LLM判断的PK: {pk_range}")
        logger.info(f"  - 职能常模: {target_profiles}")

        # pk_range 现在是单一带符号档位（如 'E+'），直接使用
        selected_pk = pk_range

        logger.info(f"  - 最终选择PK: {selected_pk}")
        logger.info(f"  - 策略说明: LLM通过两步判断法直接给出带符号的PK档位")

        # Step 3: KH收敛（基于单一PK，极致收敛）
        _ct['rules_start'] = _time.time()
        logger.info(f"[Step 3] KH收敛 - 基于单一PK: {selected_pk}")

        kh_combinations = []
        for combo_key, combo_value in self.validation_rules.kh_rules.items():
            if not isinstance(combo_key, tuple) or len(combo_key) != 3:
                continue

            pk, mk, comm = combo_key
            _, probability = combo_value

            # 只接受100%概率的组合
            if probability != '100%':
                continue

            # 检查PK是否匹配（精确匹配单一PK）
            if pk == selected_pk:
                kh_combinations.append({
                    'practical_knowledge': pk,
                    'managerial_knowledge': mk,
                    'communication': comm
                })

        logger.info(f"  - KH收敛完成: 从171个100%合法组合收敛到 {len(kh_combinations)} 个（极致收敛）")

        # Step 3.5: 智能范围扩展机制（改进版：避免策略冲突）
        if len(kh_combinations) == 0:
            logger.warning(f"初始KH收敛结果为0，启动智能范围扩展...")
            logger.warning(f"初始选择的PK: {selected_pk}")
            
            # 保存原始选择的PK
            original_selected_pk = selected_pk
            
            # 策略1：如果PK带符号，先尝试去掉符号
            if original_selected_pk.endswith(('+', '-')):
                base_pk = original_selected_pk.rstrip('+-')
                logger.info(f"策略1: 尝试去掉符号 '{original_selected_pk}' → '{base_pk}'")

                for combo_key, combo_value in self.validation_rules.kh_rules.items():
                    if not isinstance(combo_key, tuple) or len(combo_key) != 3:
                        continue
                    pk, mk, comm = combo_key
                    _, probability = combo_value

                    if probability != '100%':
                        continue

                    if pk == base_pk:
                        kh_combinations.append({
                            'practical_knowledge': pk,
                            'managerial_knowledge': mk,
                            'communication': comm
                        })

                if len(kh_combinations) > 0:
                    logger.info(f"✅ 去符号策略成功，找到 {len(kh_combinations)} 个KH组合")
                    selected_pk = base_pk  # 更新选择的PK
                else:
                    logger.info(f"去符号策略无效，继续尝试策略2")
            
            # 策略2：扩展到相邻档位（只在策略1失败时执行）
            if len(kh_combinations) == 0:
                pk_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                base_pk = original_selected_pk.rstrip('+-')  # 使用原始值

                expanded_pk_list = [base_pk]  # 包含当前档位
                if base_pk in pk_order:
                    idx = pk_order.index(base_pk)
                    if idx > 0:
                        expanded_pk_list.append(pk_order[idx - 1])  # 前一档
                    if idx < len(pk_order) - 1:
                        expanded_pk_list.append(pk_order[idx + 1])  # 后一档

                logger.info(f"策略2: 扩展到相邻档位 {expanded_pk_list}")

                for combo_key, combo_value in self.validation_rules.kh_rules.items():
                    if not isinstance(combo_key, tuple) or len(combo_key) != 3:
                        continue
                    pk, mk, comm = combo_key
                    _, probability = combo_value

                    if probability != '100%':
                        continue

                    if pk in expanded_pk_list:
                        kh_combinations.append({
                            'practical_knowledge': pk,
                            'managerial_knowledge': mk,
                            'communication': comm
                        })

                if len(kh_combinations) > 0:
                    logger.info(f"✅ 相邻档位扩展成功，找到 {len(kh_combinations)} 个KH组合")
                    # 注意：这里不更新selected_pk，因为扩展后的组合包含多个PK档位
                else:
                    logger.error(f"❌ 所有扩展策略失败，可能JD信息不足或约束过严")

        # Step 4: PS精确收敛
        ps_kh_combinations = self.converge_ps_combinations(kh_combinations)

        # Step 5: 确定Magnitude（规则驱动，基于用户输入）
        logger.info("[Step 5] 确定影响范围（Magnitude）")
        if revenue_contribution and revenue_contribution.get('type') == 'quantifiable':
            # 可量化模式：根据金额确定magnitude档位
            amount = revenue_contribution.get('amount_wan', 0)
            magnitude = determine_magnitude_from_amount(amount)
            nature_candidates = ['R', 'C', 'S', 'P']
            logger.info(f"  - 可量化模式: 金额={amount}万元 → magnitude={magnitude}")
            logger.info(f"  - Nature候选范围: RSCP")
        elif revenue_contribution and revenue_contribution.get('type') == 'not_quantifiable':
            # 不可量化模式：magnitude固定为N
            magnitude = 'N'
            nature_candidates = ['I', 'II', 'III', 'IV', 'V', 'VI']
            logger.info(f"  - 不可量化模式 → magnitude=N")
            logger.info(f"  - Nature候选范围: I-VI")
        else:
            # 默认：不可量化模式
            magnitude = 'N'
            nature_candidates = ['I', 'II', 'III', 'IV', 'V', 'VI']
            logger.info(f"  - 未指定营收贡献，默认为不可量化模式 → magnitude=N")

        # Step 6: 确定ACC因素（Freedom和Nature）- 完全基于规则推导
        logger.info("[Step 6] 确定ACC因素范围（基于规则推导）")

        # Step 6.1: 从PS+KH组合中收集所有TE档位
        all_te_values = set()
        for ps_kh_combo in ps_kh_combinations:
            te = ps_kh_combo['thinking_environment']
            all_te_values.add(te)

        logger.info(f"  - 从PS收敛结果中提取TE范围: {sorted(all_te_values)}")

        # Step 6.2: 从TE反推Freedom候选范围（基于约束链：PK ≥ TE ≥ Freedom）
        all_freedom_candidates = set()
        for te in all_te_values:
            freedom_list = self._infer_freedom_from_te(te)
            all_freedom_candidates.update(freedom_list)

        all_freedom_candidates = sorted(all_freedom_candidates)
        logger.info(f"  - 基于约束链（TE≥Freedom）反推Freedom范围: {all_freedom_candidates}")

        # Step 6.3: 从ACC规则表筛选合法的(Freedom, 'N', Nature)组合
        logger.info("  - 从ACC规则表筛选合法组合")

        # 查询所有(Freedom, 'N', Nature)的100%合法组合
        valid_acc_combinations = []
        for freedom in all_freedom_candidates:
            for nature in nature_candidates:  # I-VI
                is_valid, _, prob = self.validation_rules.validate_acc(
                    freedom, magnitude, nature
                )
                if is_valid and prob == '100%':
                    valid_acc_combinations.append((freedom, nature))

        logger.info(f"  - ACC规则表筛选: {len(all_freedom_candidates)} Freedom × {len(nature_candidates)} Nature → {len(valid_acc_combinations)} 个合法组合")

        # 提取Freedom和Nature的唯一值
        valid_freedom = list(dict.fromkeys([freedom for freedom, nature in valid_acc_combinations]))
        valid_nature = list(dict.fromkeys([nature for freedom, nature in valid_acc_combinations]))

        if not valid_freedom or not valid_nature:
            # 如果没有合法组合，使用默认值
            logger.warning("⚠️ 未找到合法的ACC组合，使用默认值")
            valid_freedom = ['C']
            valid_nature = ['III']

        acc_hints = {
            'freedom_to_act': valid_freedom,
            'nature_of_impact': valid_nature,
            'reasoning': {
                'freedom_to_act': f"基于约束链从TE反推，并通过ACC规则表筛选: {valid_freedom}",
                'nature_of_impact': f"基于ACC规则表筛选(Freedom, N, Nature)的100%合法组合: {valid_nature}"
            }
        }

        logger.info(f"✓ ACC因素确定完成（完全基于规则）")
        logger.info(f"  - Freedom: {acc_hints['freedom_to_act']}")
        logger.info(f"  - Nature: {acc_hints['nature_of_impact']}")

        # Step 7: 生成完整方案（8因素，保留规则表符号）
        logger.info("[Step 7] 生成完整方案（直接使用规则表符号）")
        all_candidates = self.generate_base_solutions_phase1(
            ps_kh_combinations,
            acc_hints,
            magnitude
        )

        logger.info(f"[Step 7] 方案生成完成 - 共 {len(all_candidates)} 个候选（直接来自规则表）")

        # Step 8: 验证并排序
        logger.info("[Step 8] 验证并排序所有候选")
        valid_solutions = self.validate_and_rank_solutions(
            all_candidates, function, target_profiles
        )

        if not valid_solutions:
            logger.warning(f"⚠️ 严格约束链下无合法方案，尝试逐步放宽条件")

            # 放宽策略1: 放宽约束链（允许相等: PK >= TE >= Freedom）
            relaxed_solutions = self._validate_with_relaxed_constraints(
                all_candidates, function, target_profiles
            )
            if relaxed_solutions:
                logger.info(f"✓ 放宽约束链后找到 {len(relaxed_solutions)} 个方案")
                valid_solutions = relaxed_solutions
            else:
                # 放宽策略2: 同时放宽约束链 + 扩展常模范围
                profile_order = ['P4', 'P3', 'P2', 'P1', 'L', 'A1', 'A2', 'A3', 'A4']
                expanded_profiles = self._expand_target_profiles(target_profiles, profile_order, expand_steps=2)
                logger.info(f"  扩展常模范围: {target_profiles} → {expanded_profiles}")

                relaxed_solutions = self._validate_with_relaxed_constraints(
                    all_candidates, function, expanded_profiles
                )
                if relaxed_solutions:
                    logger.info(f"✓ 放宽约束链+扩展常模后找到 {len(relaxed_solutions)} 个方案")
                    valid_solutions = relaxed_solutions
                else:
                    # 最终兜底: 跳过所有约束，直接计算HAY评分
                    logger.warning(f"⚠️ 所有放宽策略都未找到方案，使用无约束兜底")
                    fallback_solutions = self._validate_no_constraints(all_candidates)
                    if fallback_solutions:
                        valid_solutions = fallback_solutions
                    else:
                        return {'best_solution': None, 'match_score': 0, 'all_valid_solutions': [], 'convergence_stats': {}}

        # 使用两阶段选择策略（先Profile匹配度，再职级策略）
        best_solution, best_score = self._select_solution_by_grade_strategy(
            valid_solutions
        )

        # 学历职级兜底（本科在读≥9级，研究生在读≥10级）
        best_solution, best_score = self._apply_grade_education_floor(
            valid_solutions, best_solution, best_score, education
        )

        logger.info(f"=== 收敛完成 ===")
        logger.info(f"最优解匹配度: {best_score:.1f}")

        # Step 10: 后验推理功能已移除
        # 用户反馈：推理功能加载太慢，影响体验，已删除
        logger.info(f"[Step 10] 跳过后验推理生成（功能已移除）")

        # 不返回推理内容
        final_reasoning = {}
        final_summary = ""

        # 打印缓存统计信息
        total_lookups = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_lookups * 100) if total_lookups > 0 else 0

        logger.info(f"\n=== HAY计算缓存统计 ===")
        logger.info(f"缓存命中: {self.cache_hits} 次")
        logger.info(f"缓存未命中: {self.cache_misses} 次")
        logger.info(f"总查询次数: {total_lookups} 次")
        logger.info(f"命中率: {cache_hit_rate:.1f}%")
        if self.cache_hits > 0:
            estimated_time_saved = self.cache_hits * 15  # 假设每次缓存命中节省约15ms
            logger.info(f"预估节省时间: ~{estimated_time_saved}ms")
        logger.info(f"缓存大小: {len(self.kh_ps_cache)} 个KH/PS组合")

        # 构建详细的收敛数据（用于调试和优化分析）
        convergence_details = {
            'step1_llm_extraction': {
                'pk_range': pk_range
            },
            'step2_function_norm': {
                'function': function,
                'target_profiles': target_profiles
            },
            'step2_5_pk_selection': {
                'original_pk_range': pk_range,
                'selected_pk': selected_pk,
                'strategy': f"基于职能常模'{';'.join(sorted(target_profiles))}'的智能选择"
            },
            'step3_kh_convergence': {
                'count': len(kh_combinations),
                'combinations': [
                    {
                        'pk': combo['practical_knowledge'],
                        'mk': combo['managerial_knowledge'],
                        'comm': combo['communication']
                    }
                    for combo in kh_combinations[:100]  # 限制最多100个，避免数据过大
                ]
            },
            'step4_ps_convergence': {
                'count': len(ps_kh_combinations),
                'combinations': [
                    {
                        'pk': combo['kh_combo']['practical_knowledge'],
                        'mk': combo['kh_combo']['managerial_knowledge'],
                        'comm': combo['kh_combo']['communication'],
                        'te': combo['thinking_environment'],
                        'tc': combo['thinking_challenge']
                    }
                    for combo in ps_kh_combinations[:100]  # 限制最多100个
                ]
            },
            'step5_magnitude': {
                'magnitude': magnitude
            },
            'step6_acc_determination': {
                'freedom_candidates': acc_hints.get('freedom_to_act', []),
                'nature_candidates': acc_hints.get('nature_of_impact', [])
            },
            'step7_solution_generation': {
                'count': len(all_candidates),
                'samples': [
                    {
                        'pk': sol['practical_knowledge'],
                        'mk': sol['managerial_knowledge'],
                        'comm': sol['communication'],
                        'tc': sol['thinking_challenge'],
                        'te': sol['thinking_environment'],
                        'freedom': sol['freedom_to_act'],
                        'magnitude': sol['magnitude'],
                        'nature': sol['nature_of_impact']
                    }
                    for sol in all_candidates[:50]  # 限制最多50个样本
                ]
            },
            'step8_validation': {
                'count': len(valid_solutions),
                'top_solutions': [
                    {
                        'solution': {
                            'pk': sol[0]['practical_knowledge'],
                            'mk': sol[0]['managerial_knowledge'],
                            'comm': sol[0]['communication'],
                            'tc': sol[0]['thinking_challenge'],
                            'te': sol[0]['thinking_environment'],
                            'freedom': sol[0]['freedom_to_act'],
                            'magnitude': sol[0]['magnitude'],
                            'nature': sol[0]['nature_of_impact']
                        },
                        'match_score': sol[1],
                        'profile': sol[2],
                        'job_grade': sol[3]
                    }
                    for sol in valid_solutions[:20]  # 前20个最优解
                ]
            }
        }

        _ct['rules_end'] = _time.time()

        # 收敛引擎内部耗时汇总
        _llm_time = _ct['llm_end'] - _ct['llm_start']
        _rules_time = _ct['rules_end'] - _ct['rules_start']
        _total_conv = _ct['rules_end'] - _ct['start']
        logger.info(f"\n{'─' * 50}")
        logger.info(f"⏱️  收敛引擎内部耗时:")
        logger.info(f"  LLM调用(extract_pk): {_llm_time:.2f}s  ← {'⚠️ 瓶颈!' if _llm_time > 3 else '✓'}")
        logger.info(f"  规则收敛(Step3-8):   {_rules_time:.3f}s")
        logger.info(f"  总计:                {_total_conv:.2f}s")
        logger.info(f"{'─' * 50}")

        return {
            'best_solution': best_solution,
            'match_score': best_score,
            'all_valid_solutions': valid_solutions,
            'convergence_stats': {
                'kh_combinations': len(kh_combinations),
                'ps_kh_combinations': len(ps_kh_combinations),
                'candidates': len(all_candidates),  # 直接生成的候选数（来自规则表）
                'valid_solutions': len(valid_solutions)
            },
            'convergence_details': convergence_details,  # 新增：详细收敛数据
            'cache_stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'total_lookups': total_lookups,
                'hit_rate': f"{cache_hit_rate:.1f}%",
                'cache_size': len(self.kh_ps_cache)
            },
            'llm_reasoning': final_reasoning,  # 不再生成推理（功能已移除）
            'llm_summary': final_summary  # 不再生成总结（功能已移除）
        }


if __name__ == '__main__':
    from validation_rules import validation_rules

    engine = IncrementalConvergence(validation_rules)

    test_jd = """
    岗位名称：薪酬专员
    岗位职责：
    1. 执行公司薪酬政策和流程
    2. 维护薪酬数据，进行薪酬核算
    3. 协助薪酬调研和分析
    4. 处理员工薪酬咨询
    """

    result = engine.find_optimal_solution(
        eval_text=test_jd,
        title="薪酬专员",
        function="人力资源"
    )

    print("\n=== 最优解 ===")
    for key, value in result['best_solution'].items():
        print(f"{key}: {value}")

    print(f"\n匹配度: {result['match_score']:.1f}")
    print(f"\n收敛统计:")
    for key, value in result['convergence_stats'].items():
        print(f"  {key}: {value}")
