"""
===========================================
通用工具函数模块
===========================================

提取重复使用的工具函数，提高代码复用性
"""

from typing import List
import time
from contextlib import contextmanager
import os


# ===========================================
# 符号调整计算（统一逻辑）
# ===========================================

def extract_symbol(factor_value: str) -> str:
    """
    提取因素级别的符号 (+, -, N)

    Args:
        factor_value: 因素值，如 "A+", "B-", "C"

    Returns:
        符号字符串: '+', '-', 'N'
    """
    if factor_value.endswith('+'):
        return '+'
    elif factor_value.endswith('-'):
        return '-'
    else:
        return 'N'


def calculate_kh_symbol_adjustment(practical: str, managerial: str) -> int:
    """
    计算 Know-How 的符号调整

    规则：
    - 两个+，或一个+且无- → +2
    - 无符号，或一+一- → +1
    - 其他情况 → +0

    Args:
        practical: 实践经验/专业领域知识
        managerial: 计划、组织与整合知识

    Returns:
        调整值: 0, 1, 2
    """
    p_sym = extract_symbol(practical)
    m_sym = extract_symbol(managerial)

    symbols = [p_sym, m_sym]
    plus_count = symbols.count('+')
    minus_count = symbols.count('-')

    if plus_count == 2 or (plus_count == 1 and minus_count == 0):
        return 2
    elif (plus_count == 0 and minus_count == 0) or (plus_count == 1 and minus_count == 1):
        return 1
    else:
        return 0


def calculate_ps_symbol_adjustment(challenge: str, environment: str) -> int:
    """
    计算 Problem Solving 的符号调整

    规则：
    - 两个+ → +1
    - 一个+，没有- → +1
    - 一个+，一个- → 0（正负抵消）
    - 其他情况 → 0

    Args:
        challenge: 思考挑战
        environment: 思考环境

    Returns:
        调整值: 0, 1
    """
    c_sym = extract_symbol(challenge)
    e_sym = extract_symbol(environment)

    symbols = [c_sym, e_sym]
    plus_count = symbols.count('+')
    minus_count = symbols.count('-')

    # 有+号且没有被-号抵消
    if plus_count >= 1 and minus_count == 0:
        return 1
    # 其他情况（包括一个+一个-抵消，或全是-，或全是N）
    else:
        return 0


def calculate_acc_symbol_adjustment(freedom: str, magnitude: str, nature: str) -> int:
    """
    计算 Accountability 的符号调整

    规则：
    - 负号占主导（≥2个-，或≥1个-且无+） → +0
    - 正号占主导（≥2个+，或≥1个+且无-） → +2
    - 平衡状态 → +1

    Args:
        freedom: 行动自由度
        magnitude: 影响范围
        nature: 影响性质

    Returns:
        调整值: 0, 1, 2
    """
    f_sym = extract_symbol(freedom)
    m_sym = extract_symbol(magnitude)
    n_sym = extract_symbol(nature)

    symbols = [f_sym, m_sym, n_sym]
    plus_count = symbols.count('+')
    minus_count = symbols.count('-')

    # 负号占主导 → +0
    if minus_count >= 2:
        return 0
    if minus_count >= 1 and plus_count == 0:
        return 0

    # 正号占主导 → +2
    if plus_count >= 2:
        return 2
    if plus_count >= 1 and minus_count == 0:
        return 2

    # 平衡 → +1
    return 1


# ===========================================
# 重试策略（指数退避）
# ===========================================

def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    计算指数退避延迟时间

    Args:
        attempt: 当前重试次数（从0开始）
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）

    Returns:
        延迟时间（秒）
    """
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)


def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,),
    logger=None
):
    """
    使用指数退避策略重试函数

    Args:
        func: 要重试的函数
        max_retries: 最大重试次数
        base_delay: 基础延迟时间
        exceptions: 需要捕获的异常类型
        logger: 日志记录器

    Returns:
        函数执行结果

    Raises:
        最后一次异常
    """
    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            if attempt == max_retries - 1:
                # 最后一次重试，抛出异常
                if logger:
                    logger.error(f"重试失败，已达最大次数 {max_retries}: {e}")
                raise

            delay = exponential_backoff(attempt, base_delay)
            if logger:
                logger.warning(f"重试 {attempt + 1}/{max_retries}，{delay:.1f}秒后继续: {e}")
            time.sleep(delay)


# ===========================================
# 环境变量管理
# ===========================================

@contextmanager
def temporary_env_vars(env_vars: dict):
    """
    临时设置环境变量的上下文管理器

    Usage:
        with temporary_env_vars({'HTTP_PROXY': None, 'HTTPS_PROXY': None}):
            # 在这个块中，代理环境变量被临时移除
            make_api_call()
        # 退出后自动恢复原始值

    Args:
        env_vars: 要临时设置的环境变量字典（None表示删除）
    """
    # 保存原始值
    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)

        # 设置新值
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    try:
        yield
    finally:
        # 恢复原始值
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


# ===========================================
# 字符串处理
# ===========================================

def safe_percentage_parse(percentage_str: str) -> float:
    """
    安全解析百分比字符串为浮点数

    Args:
        percentage_str: 百分比字符串，如 "100%", " 87% "

    Returns:
        浮点数，如 1.0, 0.87
    """
    cleaned = percentage_str.strip().rstrip('%').strip()
    try:
        return float(cleaned) / 100.0
    except ValueError:
        return 0.0


def is_percentage_match(percentage_str: str, target_value: float, tolerance: float = 0.01) -> bool:
    """
    比较百分比字符串与目标值是否匹配

    Args:
        percentage_str: 百分比字符串
        target_value: 目标值（0-1之间）
        tolerance: 容差

    Returns:
        是否匹配
    """
    actual = safe_percentage_parse(percentage_str)
    return abs(actual - target_value) < tolerance
