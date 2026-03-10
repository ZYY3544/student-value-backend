"""
===========================================
通用工具函数模块
===========================================

提取重复使用的工具函数，提高代码复用性
"""

from contextlib import contextmanager
import json
import re
import os


# ===========================================
# LLM JSON 输出清洗工具
# ===========================================

def clean_json_content(content: str) -> str:
    """
    从 LLM 响应中提取纯 JSON 文本

    处理以下情况：
    1. 前面有说明文字："好的，这是结果：{...}"
    2. 包含在 Markdown 代码块中：```json{...}```
    3. 前后有空白字符或注释

    Returns:
        清洗后的纯 JSON 字符串

    Raises:
        ValueError: 无法解析为有效JSON时抛出
    """
    # 方法0：直接解析
    try:
        json.loads(content.strip())
        return content.strip()
    except json.JSONDecodeError:
        pass

    # 方法1：Markdown 代码块
    matches = re.findall(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
    if matches:
        cleaned = matches[0].strip()
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            pass

    # 方法2：第一个 { 到最后一个 }
    brace_start = content.find('{')
    brace_end = content.rfind('}')
    if brace_start != -1 and brace_end > brace_start:
        cleaned = content[brace_start:brace_end + 1]
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            pass

    # 方法3：第一个 [ 到最后一个 ]（JSON 数组）
    bracket_start = content.find('[')
    bracket_end = content.rfind(']')
    if bracket_start != -1 and bracket_end > bracket_start:
        cleaned = content[bracket_start:bracket_end + 1]
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            pass

    content_preview = content[:200] + "..." if len(content) > 200 else content
    raise ValueError(f"无法从LLM响应中提取有效JSON\n原始内容预览:\n{content_preview}")


def safe_json_parse(content: str) -> dict:
    """
    安全解析 LLM 返回的 JSON（带清洗）

    Args:
        content: LLM 返回的原始文本

    Returns:
        解析后的 dict/list

    Raises:
        ValueError: 无法解析为有效JSON
    """
    cleaned = clean_json_content(content)
    return json.loads(cleaned)


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


