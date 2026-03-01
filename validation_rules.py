"""
HAY 因素组合验证规则加载器
Loads and manages 4 layers of validation rules from CSV files
"""
import csv
from typing import Dict, List, Tuple
from pathlib import Path
from config import config


class ValidationRules:
    """加载并管理四层验证规则"""

    def __init__(self):
        """初始化并加载所有规则表"""
        # 规则字典: key = 因素组合元组, value = (结果标签, 概率字符串)
        self.kh_rules: Dict[Tuple, Tuple[str, str]] = {}
        self.ps_rules: Dict[Tuple, Tuple[str, str]] = {}
        self.acc_rules: Dict[Tuple, Tuple[str, str]] = {}
        self.ps_kh_rules: Dict[Tuple, Tuple[str, str]] = {}

        # 新增：PS百分比到(TE, TC)组合的反向索引
        self.ps_percentage_mapping: Dict[str, List[Tuple[str, str]]] = {}

        self._load_all_rules()

    def _load_csv_with_encoding(self, csv_path: Path):
        """
        尝试多种编码方式加载 CSV 文件

        Returns:
            tuple: (file_handle, csv.DictReader, encoding_used)
        """
        encodings = ['utf-8-sig', 'gbk', 'utf-8', 'gb2312', 'latin1']

        last_error = None
        for encoding in encodings:
            try:
                f = open(csv_path, 'r', encoding=encoding)
                reader = csv.DictReader(f)
                # 尝试读取第一行以验证编码是否正确
                try:
                    next(iter(reader))
                    # 重置读取位置
                    f.seek(0)
                    reader = csv.DictReader(f)
                    print(f"成功使用 {encoding} 编码打开文件: {csv_path.name}")
                    return f, reader, encoding
                except (UnicodeDecodeError, StopIteration):
                    f.close()
                    continue
            except Exception as e:
                last_error = e
                continue

        raise Exception(f"无法用任何编码打开文件 {csv_path}: {last_error}")

    
    def _load_all_rules(self):
        """加载所有四层规则"""
        # Layer 1: Know-How 内部验证 (3因素)
        self.kh_rules = self._load_kh_rules()

        # Layer 2: Problem Solving 内部验证 (2因素)
        self.ps_rules = self._load_ps_rules()

        # Layer 3: Accountability 内部验证 (3因素)
        self.acc_rules = self._load_acc_rules()

        # Layer 4: PS×KH 跨维度验证
        self.ps_kh_rules = self._load_ps_kh_rules()

        print(f"验证规则加载完成:")
        print(f"  - KH规则: {len(self.kh_rules)} 条")
        print(f"  - PS规则: {len(self.ps_rules)} 条")
        print(f"  - ACC规则: {len(self.acc_rules)} 条")
        print(f"  - PS×KH规则: {len(self.ps_kh_rules)} 条")

    def _load_kh_rules(self) -> Dict[Tuple, Tuple[str, str]]:
        """
        加载 Know-How 验证规则
        格式: (practical_knowledge, managerial_knowledge, communication) -> (结果, 概率)
        """
        rules = {}
        csv_path = Path(config.KH_VALIDATION_CSV_PATH)

        if not csv_path.exists():
            print(f"警告: KH规则文件不存在 {csv_path}")
            return rules

        try:
            f, reader, encoding = self._load_csv_with_encoding(csv_path)
            with f:
                # 直接使用中文列名（编码已自动检测）
                for row in reader:
                    key = (
                        row['实践经验/专业领域知识'].strip(),
                        row['计划、组织与整合知识'].strip(),
                        row['沟通与影响技能'].strip()
                    )
                    value = (
                        row['组合判断结果'].strip(),
                        row['组合合理性'].strip()
                    )
                    rules[key] = value
                print(f"成功加载 KH 规则: {len(rules)} 条 (编码: {encoding})")
        except Exception as e:
            print(f"加载 KH 规则失败: {e}")

        return rules

    def _load_ps_rules(self) -> Dict[Tuple, Tuple[str, str]]:
        """
        加载 Problem Solving 验证规则
        格式: (thinking_environment, thinking_challenge) -> (结果, 概率)
        同时建立：PS百分比 -> [(TE, TC), ...] 的反向索引
        """
        rules = {}
        csv_path = Path(config.PS_VALIDATION_CSV_PATH)

        if not csv_path.exists():
            print(f"警告: PS规则文件不存在 {csv_path}")
            return rules

        try:
            f, reader, encoding = self._load_csv_with_encoding(csv_path)
            with f:
                for row in reader:
                    te = row['思考环境'].strip()
                    tc = row['思考挑战'].strip()

                    key = (te, tc)
                    value = (
                        row['组合判断结果'].strip(),
                        row['组合可能性概率'].strip()
                    )
                    rules[key] = value

                    # 新增：建立PS百分比反向索引（只对100%规则建立索引）
                    if value[1] == '100%' and '对应的PS百分比' in row:
                        ps_percentage = row['对应的PS百分比'].strip()
                        if ps_percentage:  # 确保不为空
                            if ps_percentage not in self.ps_percentage_mapping:
                                self.ps_percentage_mapping[ps_percentage] = []
                            self.ps_percentage_mapping[ps_percentage].append((te, tc))

                print(f"成功加载 PS 规则: {len(rules)} 条 (编码: {encoding})")
                print(f"成功建立 PS 百分比反向索引: {len(self.ps_percentage_mapping)} 个百分比档位")
        except Exception as e:
            print(f"加载 PS 规则失败: {e}")

        return rules

    def _load_acc_rules(self) -> Dict[Tuple, Tuple[str, str]]:
        """
        加载 Accountability 验证规则
        格式: (freedom_to_act, magnitude, nature_of_impact) -> (结果, 概率)
        注意: 只有当 magnitude='N' 时才进行验证
        """
        rules = {}
        csv_path = Path(config.ACC_VALIDATION_CSV_PATH)

        if not csv_path.exists():
            print(f"警告: ACC规则文件不存在 {csv_path}")
            return rules

        try:
            f, reader, encoding = self._load_csv_with_encoding(csv_path)
            with f:
                for row in reader:
                    key = (
                        row['行动自由度'].strip(),
                        row['影响范围'].strip(),
                        row['影响性质'].strip()
                    )
                    value = (
                        row['组合判断结果'].strip(),
                        row['组合合理性'].strip()
                    )
                    rules[key] = value
                print(f"成功加载 ACC 规则: {len(rules)} 条 (编码: {encoding})")
        except Exception as e:
            print(f"加载 ACC 规则失败: {e}")

        return rules

    def _load_ps_kh_rules(self) -> Dict[Tuple, Tuple[str, str]]:
        """
        加载 PS×KH 跨维度验证规则
        格式: (ps_percentage, kh_score) -> (结果, 概率)
        """
        rules = {}
        csv_path = Path(config.PS_KH_VALIDATION_CSV_PATH)

        if not csv_path.exists():
            print(f"警告: PS×KH规则文件不存在 {csv_path}")
            return rules

        try:
            f, reader, encoding = self._load_csv_with_encoding(csv_path)
            with f:
                for row in reader:
                    # 注意: KH分数是整数
                    key = (
                        row['解决问题分数'].strip(),
                        int(row['知识技能分数'])
                    )
                    value = (
                        row['组合判断结果'].strip(),
                        row['组合合理性'].strip()
                    )
                    rules[key] = value
                print(f"成功加载 PS×KH 规则: {len(rules)} 条 (编码: {encoding})")
        except Exception as e:
            print(f"加载 PS×KH 规则失败: {e}")

        return rules

    def validate_kh(self, practical: str, managerial: str, communication: str) -> Tuple[bool, str, str]:
        """
        验证 Know-How 三因素组合

        Returns:
            (is_valid, result_label, probability)
            is_valid: 是否为合法组合（CSV中只包含100%规则，所以查到即合法）
        """
        key = (practical, managerial, communication)

        if key not in self.kh_rules:
            return False, "未知组合", "0%"

        result_label, probability = self.kh_rules[key]
        # CSV已优化为只包含100%规则，查到即为True
        return True, result_label, "100%"

    def validate_ps(self, environment: str, challenge: str) -> Tuple[bool, str, str]:
        """
        验证 Problem Solving 两因素组合

        Returns:
            (is_valid, result_label, probability)
            CSV已优化为只包含100%规则，查到即为True
        """
        key = (environment, challenge)

        if key not in self.ps_rules:
            return False, "未知组合", "0%"

        result_label, probability = self.ps_rules[key]
        # CSV已优化为只包含100%规则，查到即为True
        return True, result_label, "100%"

    def validate_acc(self, freedom: str, magnitude: str, nature: str) -> Tuple[bool, str, str]:
        """
        验证 Accountability 三因素组合

        特殊逻辑: 只有当 magnitude='N' 时才进行验证
        如果 magnitude != 'N', 直接返回 True (任何组合都合法)

        Returns:
            (is_valid, result_label, probability)
            CSV已优化为只包含100%规则，查到即为True
        """
        # 特殊逻辑: 非N时自动通过
        if magnitude != 'N':
            return True, "自动通过(magnitude≠N)", "100%"

        key = (freedom, magnitude, nature)

        if key not in self.acc_rules:
            return False, "未知组合", "0%"

        result_label, probability = self.acc_rules[key]
        # CSV已优化为只包含100%规则，查到即为True
        return True, result_label, "100%"

    def validate_ps_kh_cross(self, ps_percentage: str, kh_score: int) -> Tuple[bool, str, str]:
        """
        验证 PS×KH 跨维度匹配

        Args:
            ps_percentage: PS百分比字符串, 如 "87%"
            kh_score: KH总分 (整数)

        Returns:
            (is_valid, result_label, probability)
        """
        # print(f"\n[DEBUG] PS×KH验证:")
        # print(f"  输入: ps_percentage='{ps_percentage}', kh_score={kh_score}")
        # print(f"  类型: {type(ps_percentage)}, {type(kh_score)}")

        # 确保格式正确
        if isinstance(ps_percentage, str) and not ps_percentage.endswith('%'):
            ps_percentage = ps_percentage + '%'

        # 转换为整数
        if isinstance(kh_score, str):
            try:
                kh_score = int(kh_score)
            except:
                return False, "KH分数格式错误", "0%"

        key = (ps_percentage, kh_score)

        if key not in self.ps_kh_rules:
            return False, "未知组合", "0%"

        result_label, probability = self.ps_kh_rules[key]
        # CSV已优化为只包含100%规则，查到即为True
        return True, result_label, "100%"


# 全局单例
validation_rules = ValidationRules()
