"""
===========================================
岗位特性常模验证模块 (Profile Norm Validator)
===========================================

作用：根据职能类型验证岗位特性是否符合行业常模

验证逻辑：
- 每个职能类型都有标准的岗位特性区间（常模）
- 对比实际计算的岗位特性与常模
- 如果偏离常模，标记为需要人工复核
"""

import csv
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class ProfileNormValidator:
    """岗位特性常模验证器"""

    def __init__(self, norm_csv_path: str):
        """
        初始化验证器

        Args:
            norm_csv_path: 职能常模CSV文件路径
        """
        self.norm_csv_path = norm_csv_path
        self.function_norms: Dict[str, List[str]] = {}
        self._load_norms()

    def _load_norms(self):
        """从CSV文件加载职能常模"""
        csv_path = Path(self.norm_csv_path)

        if not csv_path.exists():
            print(f"警告: 职能常模文件不存在 {csv_path}")
            return

        # 尝试多种编码
        encodings = ['gbk', 'utf-8', 'utf-8-sig', 'gb2312']
        loaded = False

        for encoding in encodings:
            try:
                with open(csv_path, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        function_type = row['职能类型'].strip()
                        profile_range = row['岗位特性建议区间'].strip()  # 修正列名

                        # 解析岗位特性区间（支持中文分号和英文分号）
                        # 优先尝试英文分号，再尝试中文分号
                        if ';' in profile_range:
                            allowed_profiles = [p.strip() for p in profile_range.split(';') if p.strip()]
                        elif '；' in profile_range:
                            allowed_profiles = [p.strip() for p in profile_range.split('；') if p.strip()]
                        else:
                            # 单个值
                            allowed_profiles = [profile_range.strip()] if profile_range.strip() else []

                        self.function_norms[function_type] = allowed_profiles

                print(f"职能常模加载完成: {len(self.function_norms)} 个职能类型 (编码: {encoding})")
                loaded = True
                break
            except Exception as e:
                continue

        if not loaded:
            print(f"错误: 无法加载职能常模文件 {csv_path}")

    def get_all_functions(self) -> List[str]:
        """获取所有职能类型列表"""
        return sorted(self.function_norms.keys())

    def get_norm_profiles(self, function: str) -> Optional[List[str]]:
        """
        获取指定职能的标准岗位特性

        Args:
            function: 职能类型

        Returns:
            标准岗位特性列表，如果职能不存在则返回None
        """
        return self.function_norms.get(function)

    def validate_profile(self, function: str, actual_profile: str) -> Tuple[bool, str, List[str]]:
        """
        验证岗位特性是否符合职能常模

        Args:
            function: 职能类型
            actual_profile: 实际计算的岗位特性（如 A2）

        Returns:
            (is_valid, message, expected_profiles)
            - is_valid: 是否符合常模
            - message: 验证消息
            - expected_profiles: 该职能的标准岗位特性列表
        """
        # 如果职能不在常模表中，跳过验证
        if function not in self.function_norms:
            return True, f"职能'{function}'不在常模表中，跳过验证", []

        expected_profiles = self.function_norms[function]

        # 检查实际岗位特性是否在标准区间内
        if actual_profile in expected_profiles:
            return True, f"符合常模（{function}职能标准特性：{', '.join(expected_profiles)}）", expected_profiles
        else:
            # 获取实际特性的大类
            actual_category = self._get_profile_category(actual_profile)
            expected_categories = [self._get_profile_category(p) for p in expected_profiles]

            # 判断偏离程度
            if actual_category in expected_categories:
                return False, f"偏离常模但同类型（{function}职能标准特性：{', '.join(expected_profiles)}，实际：{actual_profile}）【仅供参考】", expected_profiles
            else:
                return False, f"偏离常模（{function}职能标准特性：{', '.join(expected_profiles)}，实际：{actual_profile}）【仅供参考，不影响结果有效性】", expected_profiles

    @staticmethod
    def _get_profile_category(profile: str) -> str:
        """获取岗位特性的大类（P型/L型/A型）"""
        if profile.startswith('P'):
            return 'P型'
        elif profile == 'L':
            return 'L型'
        elif profile.startswith('A'):
            return 'A型'
        else:
            return '未知'


# 全局单例（指向validation_csv文件夹的常模文件）
from config import config
profile_norm_validator = ProfileNormValidator(config.PROFILE_NORM_CSV_PATH)


# 便捷函数
def validate_job_profile(function: str, actual_profile: str) -> dict:
    """
    验证岗位特性的便捷函数

    Args:
        function: 职能类型
        actual_profile: 实际岗位特性

    Returns:
        {
            'is_valid': True/False,
            'message': '验证消息',
            'expected_profiles': ['A1', 'A2'],
            'actual_profile': 'A2',
            'function': '人力资源'
        }
    """
    is_valid, message, expected_profiles = profile_norm_validator.validate_profile(function, actual_profile)

    return {
        'is_valid': is_valid,
        'message': message,
        'expected_profiles': expected_profiles,
        'actual_profile': actual_profile,
        'function': function
    }


def get_all_function_types() -> List[str]:
    """获取所有职能类型（用于前端下拉框）"""
    functions = profile_norm_validator.get_all_functions()

    # 如果加载失败，返回备份列表
    if not functions or len(functions) == 0:
        print("⚠️ 警告：从常模表加载职能失败，使用备份列表")
        return [
            '预研', '硬件开发', '软件开发', '产品', '设计',
            '制造', '质量', '市场营销', '销售', '客户服务',
            '供应链', '人力资源', '法务', '财务', '审计',
            '战略', '行政', 'IT', '环保安防',
            '数据分析', '项目管理', '公关传播', '风控合规'
        ]

    return functions


# 示例用法
if __name__ == '__main__':
    print("=== 职能常模验证测试 ===\n")

    # 测试1：符合常模
    result = validate_job_profile('人力资源', 'A1')
    print(f"测试1 - 人力资源/A1:")
    print(f"  验证结果: {result['is_valid']}")
    print(f"  消息: {result['message']}\n")

    # 测试2：偏离常模
    result = validate_job_profile('人力资源', 'P2')
    print(f"测试2 - 人力资源/P2:")
    print(f"  验证结果: {result['is_valid']}")
    print(f"  消息: {result['message']}\n")

    # 测试3：获取所有职能
    functions = get_all_function_types()
    print(f"测试3 - 所有职能类型 ({len(functions)}个):")
    print(f"  {', '.join(functions)}")
