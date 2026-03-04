import React, { useMemo, useState, useEffect } from 'react';
import { AssessmentResult, AssessmentInput, AbilityItem } from '../types';
import {
  Sparkles, Target,
  Lightbulb, Quote, Brain, Handshake, PenTool
} from 'lucide-react';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, Tooltip
} from 'recharts';
import { ChatWidget } from './ChatWidget';

interface ResultViewProps {
  result: AssessmentResult;
  inputData: AssessmentInput;
  assessmentType: 'CV';
  onReset: () => void;
}

const ABILITY_TAGS: Record<string, Record<string, string>> = {
  专业力: { high: '实力出众', medium: '稳扎稳打', low: '积蓄力量' },
  管理力: { high: '统筹大局', medium: '团队骨干', low: '初出茅庐' },
  合作力: { high: '团队粘合', medium: '默契搭档', low: '配合协作' },
  思辨力: { high: '洞察先机', medium: '逻辑清晰', low: '按部就班' },
  创新力: { high: '开拓先锋', medium: '持续改进', low: '学习成长' },
};

const ABILITY_ICONS: Record<string, React.ReactNode> = {
  专业力: <PenTool size={20} />,
  管理力: <Target size={20} />,
  合作力: <Handshake size={20} />,
  思辨力: <Brain size={20} />,
  创新力: <Lightbulb size={20} />,
};

// CountUp 组件：数字跳动动画
const CountUp: React.FC<{ target: number; duration?: number; delay?: number; suffix?: string; prefix?: string }> = ({ target, duration = 2000, delay = 0, suffix = '', prefix = '' }) => {
  const [current, setCurrent] = useState(0);
  const [started, setStarted] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setStarted(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  useEffect(() => {
    if (!started) return;
    const startTime = performance.now();
    let rafId: number;

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // cubic ease-out
      const eased = 1 - Math.pow(1 - progress, 3);
      setCurrent(Math.round(eased * target));
      if (progress < 1) {
        rafId = requestAnimationFrame(animate);
      }
    };

    rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, [started, target, duration]);

  const display = (started ? current : 0).toString().padStart(2, '0');
  return <>{prefix}{display}{suffix}</>;
};

// 圆环进度组件
const RingChart: React.FC<{
  value: number;
  maxValue: number;
  color: string;
  gradientId: string;
  size?: number;
  strokeWidth?: number;
  label: string;
  sublabel: string;
  isPercent?: boolean;
  desc: string;
  animDelay?: number;
}> = ({ value, maxValue, color, gradientId, size = 90, strokeWidth = 8, label, sublabel, isPercent = false, desc, animDelay = 12.5 }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const percentage = Math.min(value / maxValue, 1);

  return (
    <div className="bg-white rounded-[20px] border border-gray-100 overflow-hidden flex flex-col items-center relative ring-card" style={{ animationDelay: `${animDelay - 0.7}s` }}>
      <div className="pt-4 pb-2 flex flex-col items-center w-full px-2">
        <p className="text-[13px] font-black text-gray-600 mb-2 tracking-wider">{label}</p>

        {/* SVG 圆环 */}
        <div className="relative" style={{ width: size, height: size }}>
          <svg width={size} height={size} className="transform -rotate-90">
            <defs>
              <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor={`${color}60`} />
                <stop offset="100%" stopColor={color} />
              </linearGradient>
            </defs>
            <circle
              cx={size / 2} cy={size / 2} r={radius}
              fill="none" stroke="#f1f5f9" strokeWidth={strokeWidth}
            />
            <circle
              cx={size / 2} cy={size / 2} r={radius}
              fill="none"
              stroke={`url(#${gradientId})`}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={circumference}
              className="ring-progress"
              style={{
                '--ring-target-offset': `${circumference * (1 - percentage)}`,
                '--ring-circumference': `${circumference}`,
                animationDelay: `${animDelay}s`,
              } as React.CSSProperties}
            />
          </svg>
          {/* 中心数字 */}
          <div className="absolute inset-0 flex items-center justify-center ring-center-num" style={{ animationDelay: `${animDelay + 2}s` }}>
            <span className="text-xl font-black" style={{ color }}>{isPercent ? `${value}%` : `${value}分`}</span>
          </div>
        </div>

        {/* 下方文案 */}
        <p className="text-[11px] text-gray-500 font-medium leading-relaxed mt-3 text-center ring-desc" style={{ animationDelay: `${animDelay + 2.7}s` }}>{desc}</p>
      </div>
    </div>
  );
};

// 根据百分位生成薪酬竞争力描述
function getSalaryCompDesc(val: number): string {
  if (val >= 50) return `你超过了${val}%的同届候选人，继续加油！`;
  return `你超过了${val}%的同届候选人，继续努力！`;
}

// 根据分数生成简历健康度描述
function getResumeHealthDesc(val: number): string {
  if (val >= 50) return '您的简历健康度不错，继续保持！';
  return '您的简历健康度有待提升，继续努力！';
}

export const ResultView: React.FC<ResultViewProps> = ({ result, inputData, onReset }) => {
  const theme = useMemo(() => {
    const level = result.level;
    if (level <= 14) return { bg: 'bg-[#0A66C2]', lightBg: 'bg-[#0A66C2]/10', text: 'text-[#0A66C2]', chartFill: '#0A66C2', emoji: '💎' };
    return { bg: 'bg-indigo-600', lightBg: 'bg-indigo-50', text: 'text-indigo-600', chartFill: '#4F46E5', emoji: '🚀' };
  }, [result.level]);

  const radarData = useMemo(() => {
    if (result.radarData) {
      return Object.entries(result.radarData).map(([key, val]) => ({
        subject: key, A: +(val / 10).toFixed(1)
      }));
    }
    return [
      { subject: '专业力', A: 5.0 }, { subject: '管理力', A: 5.0 },
      { subject: '合作力', A: 5.0 }, { subject: '思辨力', A: 5.0 },
      { subject: '创新力', A: 5.0 }
    ];
  }, [result.radarData]);

  const competencyDetails = useMemo(() => {
    if (!result.abilities) return [];
    return Object.entries(result.abilities).map(([name, val]) => {
      const info = val as AbilityItem;
      return {
        label: name,
        score: (info.score / 10).toFixed(1),
        tag: ABILITY_TAGS[name]?.[info.level] || info.level,
        desc: info.explanation,
        icon: ABILITY_ICONS[name] || <PenTool size={20} />,
      };
    });
  }, [result.abilities]);

  // 解析薪酬数字（如 "256k-301k" → [256, 301]）
  const salaryNumbers = useMemo(() => {
    const match = result.personValue?.match(/(\d+)\s*k?\s*[-～~]\s*(\d+)/);
    if (match) return [parseInt(match[1]), parseInt(match[2])];
    return [0, 0];
  }, [result.personValue]);

  const salaryCompetitiveness = result.salaryCompetitiveness ?? 50;
  const resumeHealthScore = result.resumeHealthScore ?? 50;

  return (
    <div className="min-h-screen bg-blue-50 flex flex-col relative pb-16 overflow-x-hidden max-w-2xl mx-auto">
      {/* Main Header Blue Section */}
      <div className="absolute top-0 left-0 right-0 h-[440px] gradient-primary rounded-b-[60px] z-0 shadow-xl opacity-95 overflow-hidden">
          <div className="absolute top-4 right-0 p-4 opacity-30 pointer-events-none">
            <div className="relative w-32 h-32 flex items-center justify-center">
               <Sparkles size={120} className="text-white opacity-90 animate-float-soft" />
            </div>
          </div>
      </div>

      <div className="relative z-10 px-6 pt-14 flex flex-col items-start text-white animate-fade-in">
          <div className="text-white/90 text-[18px] font-black mb-2 uppercase tracking-widest text-left">
            CAMPUS REPORT
          </div>
          <div className="text-4xl font-black text-white tracking-tight flex items-center gap-2 text-left">
            校招身价报告
            <Sparkles size={28} className="text-[#f8ea1a] fill-[#f8ea1a]" />
          </div>
      </div>

      <div className="relative z-10 px-5 mt-8 space-y-6 animate-fade-in">
        {/* 核心价值展示 */}
        <div className="bg-white backdrop-blur-xl rounded-[44px] p-10 border border-white flex flex-col items-center text-center relative overflow-hidden">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="absolute top-10 right-2 w-36 h-36 text-blue-300/20 select-none pointer-events-none rotate-[12deg]"><path d="M19.3788 15.1057C20.9258 11.4421 19.5373 7.11431 16.0042 5.0745C13.4511 3.60046 10.4232 3.69365 8.03452 5.0556L7.04216 3.31879C10.028 1.61639 13.8128 1.4999 17.0042 3.34245C21.4949 5.93513 23.2139 11.4848 21.1217 16.112L22.4635 16.8867L18.2984 19.1008L18.1334 14.3867L19.3788 15.1057ZM4.62961 8.89968C3.08263 12.5633 4.47116 16.8911 8.00421 18.9309C10.5573 20.4049 13.5851 20.3118 15.9737 18.9499L16.9661 20.6867C13.9803 22.389 10.1956 22.5055 7.00421 20.663C2.51357 18.0703 0.794565 12.5206 2.88672 7.89342L1.54492 7.11873L5.70999 4.90463L5.87505 9.61873L4.62961 8.89968ZM13.0042 13.5382H16.0042V15.5382H13.0042V17.5382H11.0042V15.5382H8.00421V13.5382H11.0042V12.5382H8.00421V10.5382H10.59L8.46868 8.41692L9.88289 7.00271L12.0042 9.12403L14.1255 7.00271L15.5397 8.41692L13.4184 10.5382H16.0042V12.5382H13.0042V13.5382Z"></path></svg>

            {/* 1. levelTag 移到最上面 */}
            <div className="w-full flex items-center justify-center py-2.5 mb-4 anim-delay-fade" style={{ opacity: 0, animationDelay: '9.6s' }}>
               <span className="relative inline-flex items-center">
                 <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" fill="none" className="w-7 h-7 absolute -left-9"><path d="M27.6002 18.5998V11.3998C27.6002 8.41743 25.1826 5.99977 22.2002 5.99977L15.0002 22.1998V41.9998H35.9162C37.7113 42.0201 39.2471 40.7147 39.5162 38.9398L42.0002 22.7398C42.1587 21.6955 41.8506 20.6343 41.1576 19.8373C40.4645 19.0403 39.4564 18.5878 38.4002 18.5998H27.6002Z" stroke="#f8ea1a" strokeWidth="4" strokeLinejoin="round"/><path d="M15 22.0001H10.194C8.08532 21.9628 6.2827 23.7095 6 25.7994V38.3994C6.2827 40.4894 8.08532 42.0367 10.194 41.9994H15V22.0001Z" fill="none" stroke="#f8ea1a" strokeWidth="4" strokeLinejoin="round"/></svg>
                 <h3 className="text-[#0A66C2] text-2xl font-black tracking-tight" style={{ fontFamily: "-apple-system, BlinkMacSystemFont, 'Helvetica Neue', Helvetica, Segoe UI, Arial, Roboto, 'PingFang SC', 'miui', 'Hiragino Sans GB', 'Microsoft Yahei', sans-serif" }}>"{result.levelTag}"</h3>
               </span>
            </div>

            {/* 2. 薪酬数字（CountUp 动画） */}
            <div className="mb-4 flex items-center justify-center whitespace-nowrap" style={{ transform: "scaleY(1.1)" }}>
              <span className="text-[28px] font-bold text-[#110e0c] mr-1 self-center">¥</span>
              <div className="text-[42px] font-black tracking-wide drop-shadow-sm tabular-nums text-[#110e0c] flex items-baseline">
                <span style={{ minWidth: '3ch', textAlign: 'right', display: 'inline-block' }}><CountUp target={salaryNumbers[0]} duration={3000} delay={800} /></span>
                <span className="text-[28px] ml-0.5">k</span>
                <span className="text-gray-300 mx-1">～</span>
                <span style={{ minWidth: '3ch', textAlign: 'right', display: 'inline-block' }}><CountUp target={salaryNumbers[1]} duration={3500} delay={4600} /></span>
                <span className="text-[28px] ml-0.5">k</span>
              </div>
            </div>

            {/* 3. levelDesc */}
            <div className="w-full anim-delay-scale" style={{ opacity: 0, animationDelay: '10.8s' }}>
               <p className="text-sm font-bold leading-relaxed italic text-left w-full"><span className="text-[#110e0c] opacity-80">"{result.levelDesc}"</span></p>
            </div>

            {/* 4. 两个圆环卡片 */}
            <div className="grid grid-cols-2 gap-3 w-full mt-6 -mx-2" style={{ width: 'calc(100% + 16px)' }}>
              <RingChart
                value={salaryCompetitiveness}
                maxValue={100}
                color="#0A66C2"
                gradientId="salaryGrad"
                size={80}
                strokeWidth={7}
                label="薪酬竞争力"
                sublabel="SALARY RANK"
                isPercent={true}
                desc={getSalaryCompDesc(salaryCompetitiveness)}
                animDelay={12.5}
              />
              <RingChart
                value={resumeHealthScore}
                maxValue={100}
                color="#10b981"
                gradientId="healthGrad"
                size={80}
                strokeWidth={7}
                label="简历健康度"
                sublabel="RESUME HEALTH"
                isPercent={false}
                desc={getResumeHealthDesc(resumeHealthScore)}
                animDelay={12.5}
              />
            </div>

            {/* 5. 分隔线 + 说明 */}
            <div className="w-full h-px bg-[#110e0c]/15 mt-6"></div>
            <p className="text-[11px] text-[#110e0c] opacity-30 font-medium leading-relaxed mt-6 text-left w-full">说明：以上为应届校招预估年度总薪酬包（单位：千元），由模型评估而成，仅供参考。</p>
        </div>

        {/* 核心能力画像 - 雷达图 */}
        <div className="bg-white rounded-[44px] px-8 pt-6 pb-4 shadow-sm border border-[#b7ccab]/15">
           <div className="flex items-start justify-between mb-3">
              <div className="flex flex-col">
                 <h3 className="font-black text-[#0A66C2] text-xl leading-tight">核心能力画像</h3>
                 <p className="text-[11px] text-[#110e0c] opacity-30 font-black uppercase tracking-widest mt-1 whitespace-nowrap">CORE COMPETENCY PROFILE</p>
              </div>
           </div>

           <div className="h-[280px] w-full flex items-center justify-center relative [&_*]:!outline-none">
              <ResponsiveContainer width="100%" height="100%">
                 <RadarChart cx="50%" cy="50%" outerRadius={90} data={radarData}>
                    <PolarGrid stroke="#b7ccab" opacity={0.3} />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#475569', fontSize: 13, fontWeight: 700 }} />
                    <Radar name="能力值" dataKey="A" stroke="#3b82f6" strokeWidth={3} fill="#3b82f6" fillOpacity={0.35} dot={false} activeDot={{ r: 6, fill: '#3b82f6', stroke: '#fff', strokeWidth: 2 }} />
                    <Tooltip content={({ active, payload }) => active && payload?.[0] ? <div className="bg-[#3b82f6] text-white text-xs font-black px-2.5 py-1 rounded-lg shadow">{payload[0].payload.subject}: {Number(payload[0].value).toFixed(1)}分</div> : null} />
                 </RadarChart>
              </ResponsiveContainer>
           </div>

           <div className="mt-1 text-left">
              <p className="text-[11px] text-[#110e0c] opacity-30 font-medium">
                说明：雷达图显示了您的能力得分，分值越高代表在该领域的竞争力越强。
              </p>
           </div>
        </div>

        {/* 各能力得分释义 */}
        <div className="bg-white rounded-[44px] px-8 pt-6 pb-6 shadow-sm border border-[#b7ccab]/15">
           <div className="flex flex-col mb-3">
              <h3 className="font-black text-[#0A66C2] text-xl leading-tight">各能力得分释义</h3>
              <p className="text-[11px] text-[#110e0c] opacity-30 font-black uppercase tracking-widest mt-1">SCORE DEFINITIONS</p>
           </div>

           <div className="space-y-6">
              {competencyDetails.map((item, idx) => (
                <div key={idx}>
                  <div className="flex-1 min-w-0">
                     <div className="flex items-baseline gap-2 mb-1">
                        <span className="font-black text-[#1e293b] text-base">{item.label}</span>
                        <span className="text-[#64748b] text-xs font-bold">{item.score}分</span>
                     </div>
                     <svg className="h-[6px] mb-1" style={{ width: '80px' }} viewBox="0 0 80 6"><path d="M0,3 Q5,0 10,3 T20,3 T30,3 T40,3 T50,3 T60,3 T70,3 T80,3" fill="none" stroke="#f8ea1a" strokeWidth="3.5" /></svg>
                     <p className="text-[11px] text-[#64748b] font-medium leading-relaxed"><span className="text-[#64748b]">{item.tag}:</span> {item.desc}</p>
                  </div>
                </div>
              ))}
           </div>
        </div>

        {/* 深度分析结论 */}
        <div className="bg-white rounded-[44px] px-8 pt-6 pb-6 shadow-sm border border-[#b7ccab]/15 relative overflow-hidden">
           <div className="flex flex-col mb-3">
              <h3 className="font-black text-[#0A66C2] text-xl leading-tight">深度分析结论</h3>
              <p className="text-[11px] text-[#110e0c] opacity-30 font-black uppercase tracking-widest mt-1">MODEL DEEP ANALYSIS</p>
           </div>

           <div className="bg-[#f8fafc] rounded-[24px] px-5 py-6 relative pt-10">
              <Quote className="absolute top-4 left-4 text-[#e2e8f0]" size={28} fill="currentColor" />
              <div className="text-[#334155] text-sm leading-8 font-medium text-justify space-y-4">
                {result.deepInsight
                  ? result.deepInsight.split(/\n\s*\n/).map((para, i) => (
                      <p key={i}>{para.trim().split(/(\*\*[^*]+\*\*)/).map((seg, j) =>
                        seg.startsWith('**') && seg.endsWith('**')
                          ? <span key={j} className="font-black text-[#0A66C2]" style={{ textDecoration: 'underline wavy #3b82f6', textDecorationThickness: '1.5px', textUnderlineOffset: '3px' }}>{seg.slice(2, -2)}</span>
                          : seg
                      )}</p>
                    ))
                  : <p>暂无深度评估分析</p>
                }
              </div>
           </div>
        </div>
      </div>

      {/* 再测一次按钮 */}
      <div className="px-6 mt-6 pb-16 flex relative z-10">
          <button onClick={onReset} className="w-[220px] h-[58px] bg-[#0A66C2] text-white text-2xl tracking-widest rounded-[14px] flex items-center justify-center active:scale-95 shadow-sm transition-all btn-cta-shimmer mx-auto" style={{ fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Helvetica Neue', Helvetica, Arial, sans-serif", fontWeight: 700 }}>
             再测一次
          </button>
      </div>

      {/* 简历优化助手浮窗 */}
      <ChatWidget
        assessmentContext={{
          factors: result.factors || {},
          abilities: result.abilities || {},
          grade: result.level,
          salaryRange: result.personValue || '',
          jobTitle: inputData.jobTitle,
          jobFunction: inputData.jobFunction,
          deepInsight: result.deepInsight,
        }}
        resumeText={inputData.resumeText}
        apiBase="https://student-value-backend.onrender.com"
      />
    </div>
  );
};
