export interface AssessmentInput {
  assessmentType: 'CV';
  city: string;
  industry: string;
  jobTitle: string;
  jobFunction: string;
  jobFunctions?: string[];  // 多岗位对比（最多3个）
  resumeText: string;
  resumeFile?: {
    mimeType: string;
    data: string;
  } | null;
  resumeFileName?: string;
  educationLevel: string;
  major: string;
  companyType: string;
  targetCompany: string;
}

// 单个能力维度
export interface AbilityItem {
  score: number;      // 分数 (0-100)
  level: string;      // 等级: 'high' | 'medium' | 'low'
  explanation: string; // 能力解释文案
}

// 5能力维度
export interface Abilities {
  专业力: AbilityItem;
  管理力: AbilityItem;
  合作力: AbilityItem;
  思辨力: AbilityItem;
  创新力: AbilityItem;
}

// 雷达图数据
export interface RadarData {
  专业力: number;
  管理力: number;
  合作力: number;
  思辨力: number;
  创新力: number;
}

// Part 2: 简历表达力诊断
export interface ExpressionDimension {
  score: number;        // 0-100
  level: string;        // 'high' | 'medium' | 'low'
  tip: string;          // 改进建议
}

export interface ResumeExpression {
  overallScore: number;   // 综合表达力评分 0-100
  dimensions: {
    量化程度: ExpressionDimension;
    STAR规范度: ExpressionDimension;
    信息完整度: ExpressionDimension;
    表达力度: ExpressionDimension;
    关键词覆盖: ExpressionDimension;
    结构规范度: ExpressionDimension;
  };
}

// Part 3: 岗位竞争力对比
export interface JobComparison {
  jobFunction: string;       // 岗位名称
  salaryRange: string;       // 薪酬区间
  matchScore: number;        // 匹配度 0-100
  competitiveness: number;   // 竞争力百分位 0-100
  strengths: string[];       // 核心优势维度
  gaps: string[];            // 主要差距维度
}

export interface AssessmentResult {
  // === Part 1: 能力画像 ===
  level: number;
  abilityScore?: number;           // 能力评分 0-100（新增）
  levelTag: string;                // 专业段位名称（如"独立执行者"）
  levelDesc: string;               // 段位描述
  abilities?: Abilities;
  radarData?: RadarData;
  abilitySummary?: string;
  abilityCompetitiveness?: number; // 能力百分位

  // === Part 2: 简历表达力诊断 ===
  resumeExpression?: ResumeExpression;  // 6维度表达力诊断（新增）
  resumeHealthScore?: number;           // 综合表达力评分（兼容旧字段）

  // === Part 3: 岗位竞争力对比 ===
  jobComparisons?: JobComparison[];     // 多岗位对比（新增）
  recommendedJob?: string | null;       // 推荐岗位（新增）

  // === 市场薪酬（主岗位，向后兼容） ===
  salaryRange?: string;
  marketSalary?: {
    range: string;
    note: string;
    city: string;
    industry: string;
    function: string;
  };
  salaryCompetitiveness?: number;

  // === 其他 ===
  schoolTier?: string;
  factors?: Record<string, string>;
  logId?: number;
  greeting?: string;
  resumeText?: string;

  // 旧字段（向后兼容）
  jobValue?: string;
  personValue?: string;
  currency?: string;
  strengths?: string[];
  weaknesses?: string[];
  resumeAdvice?: string;
}

export enum AppState {
  WELCOME = 'WELCOME',
  FORM = 'FORM',
  LOADING = 'LOADING',
  RESULT = 'RESULT',
}
