"""
===========================================
配置管理模块 (Configuration Module)
===========================================

作用：统一管理项目配置，支持环境变量和默认值

使用方式：
    from config import config

    glm_key = config.GLM_API_KEY
    csv_path = config.SALARY_CSV_PATH
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List

class Config:
    """配置类"""

    def __init__(self):
        # 项目根目录（config.py 现在直接在根目录下）
        self.BASE_DIR = Path(__file__).parent.absolute()

        # 尝试加载 .env 文件
        self._load_env_file()

    def _load_env_file(self):
        """加载 .env 文件"""
        env_file = self.BASE_DIR / '.env'
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # 只有当环境变量不存在时才设置
                        if key and value and not os.getenv(key):
                            os.environ[key] = value

    # ===========================================
    # 文件路径配置
    # ===========================================

    @property
    def SALARY_CSV_PATH(self) -> str:
        """薪酬数据CSV文件路径（基于P50）"""
        default_path = 'validation_csv/薪酬数据底表.csv'
        path = os.getenv('SALARY_CSV_PATH', default_path)

        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(path):
            path = str(self.BASE_DIR / path)

        return path

    @property
    def KH_VALIDATION_CSV_PATH(self) -> str:
        """Know-How 验证规则CSV文件路径"""
        default_path = 'validation_csv/knowhow报错底表.csv'
        path = os.getenv('KH_VALIDATION_CSV_PATH', default_path)

        if not os.path.isabs(path):
            path = str(self.BASE_DIR / path)

        return path

    @property
    def PS_VALIDATION_CSV_PATH(self) -> str:
        """Problem Solving 验证规则CSV文件路径"""
        default_path = 'validation_csv/PS报错底表.csv'
        path = os.getenv('PS_VALIDATION_CSV_PATH', default_path)

        if not os.path.isabs(path):
            path = str(self.BASE_DIR / path)

        return path

    @property
    def ACC_VALIDATION_CSV_PATH(self) -> str:
        """Accountability 验证规则CSV文件路径"""
        default_path = 'validation_csv/ACC报错底表.csv'
        path = os.getenv('ACC_VALIDATION_CSV_PATH', default_path)

        if not os.path.isabs(path):
            path = str(self.BASE_DIR / path)

        return path

    @property
    def PS_KH_VALIDATION_CSV_PATH(self) -> str:
        """PS×KH 跨维度验证规则CSV文件路径"""
        default_path = 'validation_csv/KH和PS报错底表.csv'
        path = os.getenv('PS_KH_VALIDATION_CSV_PATH', default_path)

        if not os.path.isabs(path):
            path = str(self.BASE_DIR / path)

        return path

    @property
    def PROFILE_NORM_CSV_PATH(self) -> str:
        """职能常模CSV文件路径"""
        default_path = 'validation_csv/职能对应岗位特性表.csv'
        path = os.getenv('PROFILE_NORM_CSV_PATH', default_path)

        if not os.path.isabs(path):
            path = str(self.BASE_DIR / path)

        return path

    # ===========================================
    # API 服务配置
    # ===========================================

    @property
    def API_HOST(self) -> str:
        """API服务主机"""
        return os.getenv('API_HOST', '0.0.0.0')

    @property
    def API_PORT(self) -> int:
        """API服务端口"""
        return int(os.getenv('API_PORT', '5000'))

    @property
    def API_DEBUG(self) -> bool:
        """是否开启调试模式"""
        return os.getenv('API_DEBUG', 'false').lower() in ('true', '1', 'yes')

    # ===========================================
    # AWS Bedrock (Sonnet) 配置
    # ===========================================

    @property
    def AWS_ACCESS_KEY_ID(self) -> Optional[str]:
        """AWS Access Key ID"""
        return os.getenv('AWS_ACCESS_KEY_ID')

    @property
    def AWS_SECRET_ACCESS_KEY(self) -> Optional[str]:
        """AWS Secret Access Key"""
        return os.getenv('AWS_SECRET_ACCESS_KEY')

    @property
    def AWS_REGION(self) -> str:
        """AWS Region"""
        return os.getenv('AWS_REGION', 'us-east-1')

    @property
    def SONNET_MODEL_ID(self) -> str:
        """AWS Bedrock Sonnet 模型 ID"""
        return os.getenv('SONNET_MODEL_ID', 'anthropic.claude-sonnet-4-20250514')

    # ===========================================
    # GLM (智谱) 配置 - 降级备用模型
    # ===========================================

    @property
    def DEEPSEEK_API_KEY(self) -> Optional[str]:
        """DeepSeek API Key（专用于 PK 档位判断）"""
        return os.getenv('DEEPSEEK_API_KEY')

    @property
    def DEEPSEEK_MODEL(self) -> str:
        """DeepSeek 模型名称"""
        return os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

    @property
    def GLM_API_KEY(self) -> Optional[str]:
        """智谱 GLM API Key"""
        return os.getenv('GLM_API_KEY')

    @property
    def GLM_MODEL(self) -> str:
        """GLM 模型名称（如 glm-4-plus、glm-4-flash）"""
        return os.getenv('GLM_MODEL', 'glm-4-plus')

    @property
    def GLM_MODEL_PLUS(self) -> str:
        """GLM 中等模型（开场白、PlanningAgent 等）"""
        return os.getenv('GLM_MODEL_PLUS', 'glm-4-plus')

    @property
    def GLM_MODEL_FLASH(self) -> str:
        """GLM 轻量模型（简历拆分等简单任务）"""
        return os.getenv('GLM_MODEL_FLASH', 'glm-4-flash')

    @property
    def GLM_BASE_URL(self) -> str:
        """GLM API 基础 URL"""
        return os.getenv('GLM_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4')

    # ===========================================
    # 用户用量预算配置
    # ===========================================

    @property
    def SONNET_BUDGET_PER_USER(self) -> float:
        """每用户 Sonnet 预算上限（人民币）"""
        return float(os.getenv('SONNET_BUDGET_PER_USER', '15.0'))

    @property
    def SONNET_INPUT_PRICE_PER_1K(self) -> float:
        """Sonnet 输入 token 价格（人民币/千token）"""
        # Claude Sonnet: ~$3/M input = ¥0.022/千token
        return float(os.getenv('SONNET_INPUT_PRICE_PER_1K', '0.022'))

    @property
    def SONNET_OUTPUT_PRICE_PER_1K(self) -> float:
        """Sonnet 输出 token 价格（人民币/千token）"""
        # Claude Sonnet: ~$15/M output = ¥0.11/千token
        return float(os.getenv('SONNET_OUTPUT_PRICE_PER_1K', '0.11'))

    # ===========================================
    # LLM 服务配置
    # ===========================================

    @property
    def LLM_TIMEOUT(self) -> int:
        """LLM请求超时时间（秒）"""
        return int(os.getenv('LLM_TIMEOUT', '120'))  # 增加到120秒

    @property
    def LLM_MAX_RETRIES(self) -> int:
        """LLM请求最大重试次数"""
        return int(os.getenv('LLM_MAX_RETRIES', '3'))

    @property
    def LLM_TEMPERATURE(self) -> float:
        """LLM温度参数（0=完全确定，0.7=创造性）"""
        return float(os.getenv('LLM_TEMPERATURE', '0.0'))  # 改为0.0确保岗位/个人能力画像完全确定

    # ===========================================
    # 验证配置
    # ===========================================

    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证配置是否完整

        返回:
            (is_valid, error_messages)
        """
        errors = []
        warnings = []

        # 检查 LLM 配置（至少需要一个可用的模型）
        has_sonnet = self.AWS_ACCESS_KEY_ID and self.AWS_SECRET_ACCESS_KEY
        has_glm = bool(self.GLM_API_KEY)

        if not has_sonnet and not has_glm:
            errors.append("至少需要配置一个 LLM 提供商（AWS Sonnet / GLM）")

        if not has_sonnet:
            warnings.append("AWS Bedrock 未配置，Sonnet 模型不可用")
        if not has_glm:
            warnings.append("GLM 未配置，主力模型不可用")

        # 检查验证规则文件（警告，不影响基本功能）
        validation_files = [
            (self.KH_VALIDATION_CSV_PATH, "KH验证规则"),
            (self.PS_VALIDATION_CSV_PATH, "PS验证规则"),
            (self.ACC_VALIDATION_CSV_PATH, "ACC验证规则"),
            (self.PS_KH_VALIDATION_CSV_PATH, "PS×KH验证规则")
        ]

        for path, name in validation_files:
            if not os.path.exists(path):
                warnings.append(f"{name}文件不存在: {path}")

        if warnings:
            print("验证规则文件缺失（将禁用自动验证功能）:")
            for warning in warnings:
                print(f"  - {warning}")

        return len(errors) == 0, errors

    def print_config(self):
        """打印当前配置（隐藏敏感信息）"""
        print("=" * 60)
        print("当前配置信息")
        print("=" * 60)
        print(f"项目根目录: {self.BASE_DIR}")
        print(f"AWS Bedrock: {'已配置' if self.AWS_ACCESS_KEY_ID else '未配置'} (Region: {self.AWS_REGION})")
        print(f"Sonnet 模型: {self.SONNET_MODEL_ID}")
        print(f"GLM API Key: {'已配置' if self.GLM_API_KEY else '未配置'}")
        print(f"GLM 模型: {self.GLM_MODEL}")
        print(f"用户 Sonnet 预算: ¥{self.SONNET_BUDGET_PER_USER}/人")
        print(f"薪酬数据文件: {self.SALARY_CSV_PATH}")
        print(f"  - 文件存在: {os.path.exists(self.SALARY_CSV_PATH)}")
        print(f"API服务地址: {self.API_HOST}:{self.API_PORT}")
        print(f"调试模式: {self.API_DEBUG}")
        print(f"LLM超时时间: {self.LLM_TIMEOUT}秒")
        print(f"LLM最大重试: {self.LLM_MAX_RETRIES}次")
        print(f"LLM温度参数: {self.LLM_TEMPERATURE}")
        print("=" * 60)

        # 验证配置
        is_valid, errors = self.validate()
        if is_valid:
            print("配置验证通过")
        else:
            print("配置验证失败:")
            for error in errors:
                print(f"  - {error}")
        print("=" * 60)


# 创建全局配置实例
config = Config()


if __name__ == '__main__':
    # 测试配置模块
    config.print_config()
