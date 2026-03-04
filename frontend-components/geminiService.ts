import { AssessmentInput, AssessmentResult, Abilities, AbilityItem } from "../types";

/**
 * 学生版校招身价评估服务
 * 调用后端 API 进行真实评估
 */

// 学生版后端 API 地址（部署后替换为 Render 地址）
const API_BASE_URL = 'https://student-value-backend.onrender.com';

// 开发模式开关：true=使用模拟数据，false=调用真实API
const USE_MOCK = false;

// API 超时时间（毫秒）
const API_TIMEOUT = 120000;

export const generateAssessment = async (input: AssessmentInput, retryCount: number = 0, pageDurations?: { welcomeS?: number; formS?: number }): Promise<AssessmentResult> => {
  // 如果是模拟模式，返回模拟数据
  if (USE_MOCK) {
    return mockAssessment(input);
  }

  // 真实 API 调用
  // 处理简历文件：如果是文件上传，需要提取文本
  let resumeText = input.resumeText || "";
  if (input.assessmentType === "CV" && input.resumeFile) {
    resumeText = `[文件: ${input.resumeFileName}]\n${input.resumeFile.data}`;
  }

  const response = await fetch(`${API_BASE_URL}/api/mini/assess`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      assessmentType: input.assessmentType,
      city: input.city,
      industry: input.industry,
      jobTitle: input.jobTitle,
      jobFunction: input.jobFunction,
      resumeText: resumeText,
      educationLevel: input.educationLevel,
      major: input.major,
      companyType: input.companyType,
      targetCompany: input.targetCompany,
      inviteCode: localStorage.getItem('invite_code') || '',
      retryCount: retryCount,
      welcomeS: pageDurations?.welcomeS,
      formS: pageDurations?.formS,
    }),
    signal: AbortSignal.timeout(API_TIMEOUT),
  });

  if (!response.ok) {
    let errorMsg = `服务器错误 (${response.status})`;
    try {
      const errorData = await response.json();
      errorMsg = errorData.error || errorMsg;
    } catch {}
    if (response.status === 422 && errorMsg === 'insufficient_input') {
      throw new Error('insufficient_input');
    }
    if (response.status === 403) {
      throw new Error(`403: ${errorMsg}`);
    }
    throw new Error(errorMsg);
  }

  const result = await response.json();

  if (!result.success) {
    throw new Error(result.error || "评估失败");
  }

  const data = result.data;
  if (!data || typeof data !== "object") {
    throw new Error("服务器返回数据格式异常");
  }

  if (typeof data.level !== "number" || !data.salaryRange) {
    throw new Error("服务器返回数据不完整");
  }

  // 将后端返回的数据转换为前端期望的格式
  return {
    // 薪酬信息
    jobValue: data.salaryRange,
    personValue: data.salaryRange,
    currency: "人民币",

    // 职级和标签
    level: data.level,
    levelTag: data.levelTag,
    levelDesc: data.levelDesc,

    // 能力分析（付费内容）
    abilities: data.abilities,
    radarData: data.radarData,
    abilitySummary: data.abilitySummary,
    deepInsight: data.deepInsight,
    salaryCompetitiveness: data.salaryCompetitiveness,
    resumeHealthScore: data.resumeHealthScore,

    // HAY 8因素（用于简历优化助手）
    factors: data.factors,

    // 生成优劣势（基于能力数据）
    strengths: generateStrengths(data.abilities, input),
    weaknesses: generateWeaknesses(data.abilities),
    resumeAdvice: generateAdvice(data.abilities, input.jobTitle),

    logId: data.logId,
  };
};

/**
 * 对能力数据按 score 排序
 */
function sortAbilitiesByScore(abilities: Abilities, desc = true): [string, AbilityItem][] {
  return Object.entries(abilities)
    .sort(([, a], [, b]) => desc ? b.score - a.score : a.score - b.score);
}

/**
 * 根据能力数据生成优势描述
 */
function generateStrengths(
  abilities: Abilities | undefined,
  input: AssessmentInput
): string[] {
  if (!abilities) {
    return [
      `深耕${input.industry || "行业"}领域，具备极强的业务闭环思维`,
      `${input.jobFunction || "职能"}核心技术栈积淀深厚`,
      "具备处理复杂组织协调与跨部门推动的能力",
    ];
  }

  const strengths: string[] = [];
  const sorted = sortAbilitiesByScore(abilities, true);

  for (let i = 0; i < Math.min(2, sorted.length); i++) {
    const [, info] = sorted[i];
    if (info.score >= 60 && info.explanation) {
      strengths.push(info.explanation);
    }
  }

  if (strengths.length < 3) {
    strengths.push(`在${input.industry || "行业"}领域具备扎实的专业基础`);
  }

  return strengths.slice(0, 3);
}

/**
 * 根据能力数据生成劣势/提升建议
 */
function generateWeaknesses(abilities: Abilities | undefined): string[] {
  if (!abilities) {
    return [
      "在全球化视野及多元文化团队管理上仍有提升空间",
      "在极速增长业务下的系统性风险防控经验略显薄弱",
    ];
  }

  const weaknesses: string[] = [];
  const sorted = sortAbilitiesByScore(abilities, false);

  for (let i = 0; i < Math.min(2, sorted.length); i++) {
    const [name, info] = sorted[i];
    if (info.score < 60) {
      weaknesses.push(`【${name}】方面仍有提升空间：${getImprovementSuggestion(name)}`);
    }
  }

  if (weaknesses.length === 0) {
    weaknesses.push("整体能力均衡，建议在某一领域持续深耕形成核心竞争力");
  }

  return weaknesses.slice(0, 2);
}

/**
 * 获取能力提升建议
 */
function getImprovementSuggestion(abilityName: string): string {
  const suggestions: Record<string, string> = {
    专业力: "建议持续学习行业前沿知识，考取相关专业认证",
    管理力: "建议主动承担项目管理职责，积累团队协调经验",
    合作力: "建议加强跨部门沟通，拓展职场人脉网络",
    思辨力: "建议多参与复杂问题分析，培养系统性思维",
    创新力: "建议关注行业新趋势，尝试新的工作方法",
  };
  return suggestions[abilityName] || "建议持续提升";
}

/**
 * 生成简历建议
 */
function generateAdvice(abilities: Abilities | undefined, jobTitle: string): string {
  if (!abilities) {
    return `针对您的 ${jobTitle} 角色，建议在简历中进一步量化核心业务指标的增长贡献。`;
  }

  const sorted = sortAbilitiesByScore(abilities, true);
  if (sorted.length > 0) {
    const [name] = sorted[0];
    return `您的【${name}】是核心优势，建议在简历和面试中重点突出相关成就，用数据量化您的贡献。`;
  }

  return `建议在简历中突出您在 ${jobTitle} 岗位上的核心成就。`;
}

/**
 * 模拟评估（仅开发调试时使用）
 */
async function mockAssessment(input: AssessmentInput): Promise<AssessmentResult> {
  await new Promise((resolve) => setTimeout(resolve, 2000));

  const city = input.city || "上海";
  const jobTitle = input.jobTitle || "资深专家";

  let baseSalary = 20;
  if (["上海", "北京", "深圳"].includes(city)) baseSalary = 30;
  else if (["杭州", "广州"].includes(city)) baseSalary = 25;

  const mockLevel = Math.floor(Math.random() * 5) + 12;
  const minVal = baseSalary + Math.floor(Math.random() * 10);
  const maxVal = minVal + 15 + Math.floor(Math.random() * 10);

  const tags = ["搬砖圣手", "业务脊梁", "六边形战士", "职场卷王", "行业清流"];
  const descs = [
    "手在动，脑在算，砖不一定是你的，但 KPI 是。",
    "你是团队的中流砥柱，也是老板深夜加班时唯一想到的挡箭牌。",
    "你的简历在猎头圈已经传疯了，只是你自己还不知道。",
    "在内卷的浪潮中，你优雅地划着水，却总能第一个上岸～",
  ];

  const mockAbilities: Abilities = {
    专业力: { score: 60 + Math.floor(Math.random() * 25), level: "medium", explanation: "具备扎实的专业基础，能够独立完成常规专业工作" },
    管理力: { score: 50 + Math.floor(Math.random() * 25), level: "medium", explanation: "能够管理自己的工作任务，配合团队完成目标" },
    合作力: { score: 55 + Math.floor(Math.random() * 25), level: "medium", explanation: "能够在团队内部有效沟通，配合完成协作任务" },
    思辨力: { score: 55 + Math.floor(Math.random() * 25), level: "medium", explanation: "能够按照既定框架分析和解决问题，逻辑清晰" },
    创新力: { score: 50 + Math.floor(Math.random() * 25), level: "medium", explanation: "能够在现有框架下完成工作，适应变化" },
  };

  return {
    level: mockLevel,
    levelTag: tags[Math.floor(Math.random() * tags.length)],
    levelDesc: descs[Math.floor(Math.random() * descs.length)],
    jobValue: `${minVal - 2}万-${maxVal - 5}万`,
    personValue: `${minVal}万-${maxVal}万`,
    currency: "人民币",
    abilities: mockAbilities,
    radarData: {
      专业力: mockAbilities.专业力.score,
      管理力: mockAbilities.管理力.score,
      合作力: mockAbilities.合作力.score,
      思辨力: mockAbilities.思辨力.score,
      创新力: mockAbilities.创新力.score,
    },
    abilitySummary: "您的核心优势在于专业力，建议持续深耕专业领域。",
    deepInsight: `基于对您${input.jobTitle || "岗位"}履历的综合分析，您在**专业领域**有较为扎实的积累，尤其在${input.industry || "所在行业"}的**实操经验**方面表现突出。建议在未来的职业发展中，进一步强化**管理能力**和**跨领域协作能力**，以实现从专业型人才向**复合型人才**的转变。同时，关注**行业前沿趋势**，持续提升**创新思维**，将有助于在竞争激烈的市场中保持优势。`,
    strengths: [
      `深耕${input.industry || "行业"}领域，具备极强的业务闭环思维`,
      `${input.jobFunction || "职能"}核心技术栈积淀深厚`,
      "具备处理复杂组织协调与跨部门推动的能力",
    ],
    weaknesses: [
      "在全球化视野及多元文化团队管理上仍有提升空间",
      "在极速增长业务下的系统性风险防控经验略显薄弱",
    ],
    resumeAdvice: `针对您的 ${jobTitle} 角色，建议在后续报告中进一步量化核心业务指标的增长因果。`,
  };
}
