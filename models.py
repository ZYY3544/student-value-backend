from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional, Any, Union
from enums import (
    PracticalKnowledge,
    ManagerialKnowledge,
    Communication,
    ThinkingChallenge,
    ThinkingEnvironment,
    FreedomToAct,
    Magnitude,
    NatureOfImpactQuantified,
    NatureOfImpactNonQuantified
)


class HayFactors(BaseModel):
    """
    HAY评估八因素模型（带枚举约束）

    所有因素值都必须在预定义的有效值范围内
    """
    practical_knowledge: PracticalKnowledge
    managerial_knowledge: ManagerialKnowledge
    communication: Communication
    thinking_challenge: ThinkingChallenge
    thinking_environment: ThinkingEnvironment
    freedom_to_act: FreedomToAct
    magnitude: Magnitude
    # nature_of_impact 需要根据 magnitude 是否为 'N' 来决定类型
    nature_of_impact: str  # 暂时保持str类型，用validator验证
    reasoning: Optional[Dict[str, str]] = Field(default=None, description="LLM的推理过程")

    @field_validator('nature_of_impact')
    @classmethod
    def validate_nature_of_impact(cls, v: str, info) -> str:
        """
        验证nature_of_impact的值
        根据magnitude是否为'N'选择不同的有效值范围
        """
        # 自动纠正LLM常见的越界输出
        NI_AUTO_CORRECT = {
            'VI': 'V+', 'VI-': 'V+', 'VI+': 'V+',
            'VII': 'V+', '6': 'V+',
            '0': 'I-', '0-': 'I-',
        }
        if v in NI_AUTO_CORRECT:
            v = NI_AUTO_CORRECT[v]

        # 获取magnitude的值
        magnitude = info.data.get('magnitude')

        # 根据magnitude选择对应的枚举类
        if magnitude == 'N':
            # 不可量化模式
            valid_values = [e.value for e in NatureOfImpactNonQuantified]
        else:
            # 可量化模式
            valid_values = [e.value for e in NatureOfImpactQuantified]

        if v not in valid_values:
            raise ValueError(
                f"nature_of_impact值'{v}'无效。"
                f"当magnitude={'N' if magnitude == 'N' else '可量化'}时，"
                f"有效值为: {valid_values}"
            )

        return v

    class Config:
        # 允许使用枚举的值或名称
        use_enum_values = True

class JobProfile(BaseModel):
    """岗位特性（Short Profile）"""
    profile_type: str           # P4/P3/P2/P1/L/A1/A2/A3/A4
    profile_category: str       # P型/L型/A型
    level_gap: int              # PS Level与ACC Level的差距
    description: str            # 岗位特性描述

class HaySummary(BaseModel):
    kh_level: int
    kh_score: int
    ps_base_level: int
    ps_percentage: str
    ps_level: int
    ps_score: int
    acc_level: int
    acc_score: int
    total_score: int
    job_grade: int
    job_profile: Optional[JobProfile] = None  # 岗位特性

class KnowHowResult(BaseModel):
    practical_score: int
    managerial_score: int
    communication_score: int
    base_level: int
    symbol_adjustment: int
    kh_level: int
    kh_score: int

class ProblemSolvingResult(BaseModel):
    challenge_score: int
    environment_score: int
    base_level: int
    symbol_adjustment: int
    ps_base_level: int
    ps_percentage: float
    ps_percentage_str: str
    ps_level: int
    ps_score: int

class AccountabilityResult(BaseModel):
    freedom_score: int
    magnitude_score: int
    nature_score: int
    base_level: int
    symbol_adjustment: int
    acc_level: int
    acc_score: int
    is_quantified: bool

class HayResult(BaseModel):
    input_factors: HayFactors
    know_how: KnowHowResult
    problem_solving: ProblemSolvingResult
    accountability: AccountabilityResult
    total_score: int
    job_grade: int
    summary: HaySummary