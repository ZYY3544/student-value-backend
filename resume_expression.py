"""
===========================================
简历表达力诊断模块 (Resume Expression Diagnosis)
===========================================
评估简历的写作质量，与 HAY 能力评估完全独立。

6个维度：
1. 量化程度 - bullet point 中包含数字/数据的比例
2. STAR规范度 - 经历描述是否符合 Situation-Task-Action-Result 结构
3. 信息完整度 - 经历描述是否有足够的上下文和细节
4. 表达力度 - 动词是否有力，是否主动语态
5. 关键词覆盖 - 与目标岗位相关的术语覆盖程度
6. 结构规范度 - 简历整体结构、分区、时间线是否清晰
"""

import re
import json
from typing import Dict, Optional
from logger import get_module_logger

logger = get_module_logger(__name__)


# ===========================================
# 规则引擎：基于文本分析的快速评估
# ===========================================

# 弱动词列表（表达力度低的词汇）
WEAK_VERBS = {
    '参与', '参加', '协助', '帮助', '配合', '辅助',
    '了解', '学习', '接触', '熟悉',
    '负责', '完成', '处理', '执行',
}

# 强动词列表（表达力度高的词汇）
STRONG_VERBS = {
    '主导', '设计', '搭建', '构建', '开发', '创建',
    '推动', '引领', '策划', '统筹', '组织', '带领',
    '优化', '提升', '改进', '重构', '迭代',
    '独立', '从零', '落地', '交付', '实现',
    '分析', '洞察', '诊断', '评估', '制定',
}

# 各岗位关键词库（用于关键词覆盖度评估）
JOB_FUNCTION_KEYWORDS = {
    "软件开发": {"开发", "编程", "代码", "架构", "API", "数据库", "前端", "后端", "微服务", "部署", "测试", "Git", "敏捷", "需求", "性能"},
    "产品管理": {"需求", "用户", "产品", "迭代", "MVP", "调研", "竞品", "PRD", "功能", "上线", "数据", "转化", "留存", "增长", "体验"},
    "数据分析与商业智能": {"数据", "分析", "SQL", "Python", "报表", "可视化", "指标", "建模", "洞察", "BI", "ETL", "A/B测试", "漏斗", "趋势", "预测"},
    "算法": {"算法", "模型", "机器学习", "深度学习", "训练", "推理", "特征", "NLP", "CV", "优化", "精度", "召回", "AUC", "数据集", "论文"},
    "人力资源": {"招聘", "培训", "绩效", "薪酬", "组织", "人才", "HRBP", "员工", "企业文化", "团队", "面试", "入职", "人力", "劳动", "激励"},
    "市场营销": {"营销", "品牌", "推广", "投放", "获客", "转化", "ROI", "渠道", "内容", "活动", "传播", "媒体", "曝光", "流量", "增长"},
    "财务管理": {"财务", "预算", "成本", "审计", "报表", "核算", "税务", "资金", "利润", "合规", "风控", "会计", "账务", "收入", "费用"},
    "产品运营": {"运营", "用户", "活跃", "留存", "增长", "社区", "内容", "活动", "数据", "转化", "拉新", "促活", "策略", "复购", "GMV"},
    "项目管理": {"项目", "进度", "里程碑", "风险", "资源", "协调", "交付", "排期", "跟踪", "需求", "干系人", "管理", "计划", "迭代", "复盘"},
    "销售": {"销售", "客户", "业绩", "签约", "合同", "商机", "拜访", "谈判", "达成", "目标", "KPI", "渠道", "回款", "复购", "关系"},
}


def _count_quantified_bullets(text: str) -> tuple:
    """统计包含量化数据的要点数量"""
    # 按换行分割，过滤空行
    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 10]
    if not lines:
        return 0, 0

    quantified = 0
    # 匹配数字相关的模式：百分比、倍数、数字+单位
    number_pattern = re.compile(r'\d+[\.\d]*\s*[%％万亿元人次个件倍kKwW]|(?:增长|提升|降低|减少|达到|超过|完成|覆盖).*?\d+|第[一二三]|TOP\s*\d+|\d{2,}', re.IGNORECASE)

    for line in lines:
        if number_pattern.search(line):
            quantified += 1

    return quantified, len(lines)


def _evaluate_star_completeness(text: str) -> int:
    """评估 STAR 法则的完整度 (0-100)"""
    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 15]
    if not lines:
        return 30

    scores = []
    for line in lines:
        score = 0
        # Situation: 有背景描述
        if any(kw in line for kw in ['背景', '期间', '项目', '团队', '公司', '部门', '实习', '负责']):
            score += 25
        # Task: 有任务/目标描述
        if any(kw in line for kw in ['目标', '任务', '需求', '负责', '承担', '职责']):
            score += 25
        # Action: 有行动描述
        if any(kw in line for kw in ['通过', '采用', '使用', '运用', '设计', '搭建', '开发', '推动', '制定', '执行', '优化', '建立']):
            score += 25
        # Result: 有结果描述
        if any(kw in line for kw in ['提升', '增长', '降低', '减少', '达成', '完成', '实现', '获得', '产出', '效果', '成果', '%', '倍']):
            score += 25
        scores.append(score)

    if not scores:
        return 30
    return max(30, min(95, int(sum(scores) / len(scores))))


def _evaluate_verb_strength(text: str) -> int:
    """评估动词力度 (0-100)"""
    weak_count = sum(1 for verb in WEAK_VERBS if verb in text)
    strong_count = sum(1 for verb in STRONG_VERBS if verb in text)

    total = weak_count + strong_count
    if total == 0:
        return 50  # 无法判断

    strength_ratio = strong_count / total
    # 映射到 30-95 区间
    return max(30, min(95, int(30 + strength_ratio * 65)))


def _evaluate_keyword_coverage(text: str, job_function: str) -> int:
    """评估岗位关键词覆盖度 (0-100)"""
    keywords = JOB_FUNCTION_KEYWORDS.get(job_function)
    if not keywords:
        # 对于没有预定义关键词的岗位，使用通用评估
        # 检查是否有行业术语、专业词汇等
        professional_indicators = len(re.findall(r'[A-Z]{2,}|[\u4e00-\u9fff]{2}(?:管理|分析|运营|开发|设计|策划)', text))
        return max(30, min(85, 40 + professional_indicators * 3))

    matched = sum(1 for kw in keywords if kw in text)
    coverage = matched / len(keywords)
    return max(20, min(95, int(20 + coverage * 75)))


def _evaluate_structure(text: str) -> int:
    """评估简历结构规范度 (0-100)"""
    score = 50  # 基础分

    # 检查是否有分区标志
    section_markers = ['教育', '实习', '项目', '经历', '技能', '校园', '工作', '荣誉', '获奖', '证书', '自我']
    sections_found = sum(1 for marker in section_markers if marker in text)
    score += min(20, sections_found * 5)

    # 检查时间线格式
    time_patterns = len(re.findall(r'20\d{2}[\.\-/年]', text))
    if time_patterns >= 2:
        score += 10
    elif time_patterns >= 1:
        score += 5

    # 检查长度合理性（太短或太长都不好）
    text_len = len(text)
    if 300 <= text_len <= 3000:
        score += 10
    elif text_len < 150:
        score -= 10

    # 检查是否有合理的分段
    paragraphs = [p for p in text.split('\n') if p.strip()]
    if 10 <= len(paragraphs) <= 50:
        score += 5

    return max(20, min(95, score))


def _evaluate_information_completeness(text: str) -> int:
    """评估信息完整度 (0-100)"""
    score = 40  # 基础分

    # 检查基本信息完整性
    if re.search(r'教育|学历|学校|大学|学院', text):
        score += 10
    if re.search(r'实习|工作经[历验]|项目经[历验]', text):
        score += 10
    if re.search(r'技能|技术|工具|语言|框架', text):
        score += 8
    if re.search(r'获奖|荣誉|证书|奖学金', text):
        score += 5

    # 检查经历描述的细节程度
    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 10]
    detailed_lines = sum(1 for l in lines if len(l) > 30)
    detail_ratio = detailed_lines / max(len(lines), 1)
    score += int(detail_ratio * 20)

    # 检查是否有具体的公司/组织名称
    if re.search(r'[\u4e00-\u9fff]{2,}(?:公司|集团|科技|有限|股份|银行|基金|证券)', text):
        score += 7

    return max(20, min(95, score))


def evaluate_resume_expression(
    resume_text: str,
    job_function: str = "通用职能",
) -> Dict:
    """
    评估简历表达力（6个维度）

    Args:
        resume_text: 简历文本
        job_function: 目标岗位职能

    Returns:
        {
            "overallScore": 62,
            "dimensions": {
                "量化程度": {"score": 35, "level": "low", "tip": "..."},
                "STAR规范度": {"score": 72, "level": "medium", "tip": "..."},
                ...
            }
        }
    """
    logger.info(f"[ResumeExpression] 开始评估简历表达力，目标职能={job_function}")

    if not resume_text or len(resume_text.strip()) < 50:
        return _default_expression_result()

    # 1. 量化程度
    quantified, total_bullets = _count_quantified_bullets(resume_text)
    quant_ratio = quantified / max(total_bullets, 1)
    quant_score = max(20, min(95, int(20 + quant_ratio * 75)))

    # 2. STAR规范度
    star_score = _evaluate_star_completeness(resume_text)

    # 3. 信息完整度
    completeness_score = _evaluate_information_completeness(resume_text)

    # 4. 表达力度
    verb_score = _evaluate_verb_strength(resume_text)

    # 5. 关键词覆盖
    keyword_score = _evaluate_keyword_coverage(resume_text, job_function)

    # 6. 结构规范度
    structure_score = _evaluate_structure(resume_text)

    dimensions = {
        "量化程度": {
            "score": quant_score,
            "level": _score_to_level(quant_score),
            "tip": _get_quant_tip(quant_score, quantified, total_bullets),
        },
        "STAR规范度": {
            "score": star_score,
            "level": _score_to_level(star_score),
            "tip": _get_star_tip(star_score),
        },
        "信息完整度": {
            "score": completeness_score,
            "level": _score_to_level(completeness_score),
            "tip": _get_completeness_tip(completeness_score),
        },
        "表达力度": {
            "score": verb_score,
            "level": _score_to_level(verb_score),
            "tip": _get_verb_tip(verb_score),
        },
        "关键词覆盖": {
            "score": keyword_score,
            "level": _score_to_level(keyword_score),
            "tip": _get_keyword_tip(keyword_score, job_function),
        },
        "结构规范度": {
            "score": structure_score,
            "level": _score_to_level(structure_score),
            "tip": _get_structure_tip(structure_score),
        },
    }

    # 综合评分：加权平均（量化和STAR权重更高）
    weights = {
        "量化程度": 0.22,
        "STAR规范度": 0.22,
        "信息完整度": 0.15,
        "表达力度": 0.15,
        "关键词覆盖": 0.14,
        "结构规范度": 0.12,
    }
    overall = int(sum(dimensions[k]["score"] * weights[k] for k in dimensions))
    overall = max(10, min(95, overall))

    logger.info(f"[ResumeExpression] 评估完成: 综合={overall}, "
                + ", ".join(f"{k}={v['score']}" for k, v in dimensions.items()))

    return {
        "overallScore": overall,
        "dimensions": dimensions,
    }


def _score_to_level(score: int) -> str:
    if score >= 70:
        return "high"
    elif score >= 45:
        return "medium"
    else:
        return "low"


def _get_quant_tip(score: int, quantified: int, total: int) -> str:
    if score >= 70:
        return f"不错！{total}条描述中有{quantified}条包含量化数据，继续保持。"
    elif score >= 45:
        return f"{total}条描述中仅{quantified}条有数据支撑，建议为更多经历补充具体数字。"
    else:
        return "量化数据严重不足，建议在每段经历中加入具体数据（如人数、金额、百分比、时长等）。"


def _get_star_tip(score: int) -> str:
    if score >= 70:
        return "经历描述结构较好，大部分包含了背景、任务、行动和结果。"
    elif score >= 45:
        return "部分经历描述缺少结果导向，建议补充具体行动和可衡量的成果。"
    else:
        return "经历描述偏笼统，建议用STAR法则重写：交代背景、明确任务、描述行动、展示结果。"


def _get_completeness_tip(score: int) -> str:
    if score >= 70:
        return "简历信息较为完整，包含了必要的教育、经历和技能信息。"
    elif score >= 45:
        return "部分模块信息不够充实，建议补充技能清单或项目细节。"
    else:
        return "简历信息较为单薄，建议补充完整的教育背景、实习/项目经历、技能清单等。"


def _get_verb_tip(score: int) -> str:
    if score >= 70:
        return "动词使用有力，表达主动且有影响力。"
    elif score >= 45:
        return "部分描述用词偏弱（如'参与''协助'），建议替换为更有力的动词（如'主导''推动''搭建'）。"
    else:
        return "大量使用'参与''协助'等被动弱动词，建议改为'主导''设计''推动'等主动强动词。"


def _get_keyword_tip(score: int, job_function: str) -> str:
    if score >= 70:
        return f"与{job_function}岗位的关键术语匹配度较高。"
    elif score >= 45:
        return f"与{job_function}岗位有一定匹配，建议增加更多该领域的专业术语和工具名称。"
    else:
        return f"与{job_function}岗位的关键词匹配度较低，建议参考目标JD补充相关术语。"


def _get_structure_tip(score: int) -> str:
    if score >= 70:
        return "简历结构清晰，分区合理，时间线明确。"
    elif score >= 45:
        return "结构基本合理，建议完善时间线标注和模块分区。"
    else:
        return "简历结构不够清晰，建议按'教育-实习-项目-技能'分区组织，每段注明时间。"


def _default_expression_result() -> Dict:
    """简历内容不足时的默认结果"""
    return {
        "overallScore": 25,
        "dimensions": {
            "量化程度": {"score": 20, "level": "low", "tip": "简历内容过少，无法评估量化程度。"},
            "STAR规范度": {"score": 20, "level": "low", "tip": "简历内容过少，无法评估STAR结构。"},
            "信息完整度": {"score": 20, "level": "low", "tip": "简历信息严重不足，请补充完整内容。"},
            "表达力度": {"score": 25, "level": "low", "tip": "简历内容过少，无法评估表达力度。"},
            "关键词覆盖": {"score": 20, "level": "low", "tip": "简历内容过少，无法评估关键词覆盖。"},
            "结构规范度": {"score": 25, "level": "low", "tip": "简历内容过少，无法评估结构。"},
        },
    }


if __name__ == "__main__":
    # 测试用例
    test_resume = """
    教育背景
    清华大学 | 计算机科学与技术 | 本科 | 2021.09 - 2025.06

    实习经历
    字节跳动 | 产品经理实习生 | 2024.06 - 2024.09
    - 主导了用户增长策略的调研和方案设计，通过A/B测试验证3种增长方案，最终方案使日活提升12%
    - 协助团队完成了产品需求文档的撰写，参与了2个版本的迭代
    - 负责用户反馈收集和分析，建立了用户反馈分类体系

    腾讯 | 数据分析实习生 | 2023.07 - 2023.09
    - 使用SQL和Python对用户行为数据进行分析，产出周报和月报
    - 参与了推荐算法效果评估，完成了5个A/B实验的数据分析

    项目经历
    校园二手交易平台 | 项目负责人 | 2023.03 - 2023.06
    - 从零搭建了基于微信小程序的校园二手交易平台，覆盖3000+用户
    - 设计了商品推荐算法，使用户浏览转化率提升25%

    技能
    Python, SQL, Excel, Tableau, Axure, Figma
    """

    result = evaluate_resume_expression(test_resume, "产品管理")
    print(f"\n综合评分: {result['overallScore']}/100")
    print("-" * 40)
    for name, info in result['dimensions'].items():
        bar = '█' * (info['score'] // 10) + '░' * (10 - info['score'] // 10)
        print(f"  {name:　<6} {bar} {info['score']:3d}  [{info['level']}]")
        print(f"         {info['tip']}")
