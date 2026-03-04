# PRD：多 Agent 简历优化系统

> 版本：v1.0
> 日期：2026-03-04
> 状态：已实现（基础架构）
> 关联文档：[产品规划](product-plan-resume-optimization.md) | [设计探讨](conversation-agent-design-deep-dive.md)

---

## 一、背景与动机

### 1.1 问题陈述

在前期实现中，简历优化 Agent 采用**单 Agent 架构**：一个 ChatAgent 承担所有职责（诊断、优化、总结、阶段控制），使用一个巨长的 System Prompt 来约束行为。

随着功能迭代，单 Agent 架构暴露出两个根本矛盾：

| 矛盾 | 表现 |
|---|---|
| **质量 vs Prompt 长度** | 要提升某个阶段的输出质量，就要在 Prompt 中加更多规则和示例，但 Prompt 越长，各阶段指令互相稀释，整体质量反而下降 |
| **稳定性 vs 上下文长度** | 多轮对话后上下文膨胀，即使有首尾强化和对话压缩，仍会出现角色漂移和指令遗忘 |

**核心结论**：这不是 Prompt 工程的问题，而是单 Agent 架构的物理极限。

### 1.2 目标

以**用户体验最优**为唯一目标（不考虑成本），同时实现：

1. **极致的问答质量**：每个环节的输出都达到最高水准
2. **高稳定性**：无论对话多长，行为始终一致，不出现角色漂移

### 1.3 解决方案

将单 Agent 拆分为**协调者 + 多个专精子 Agent**的多 Agent 架构。

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端（不变）                              │
│  POST /api/chat/start → POST /api/chat/message (SSE) → history  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     mini_api.py（不变）                          │
│  chat_start() → chat_message() → chat_history()                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                                                                  │
│          ChatAgent（协调者 / Orchestrator）                       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  职责（极简）：                                          │    │
│  │  1. 会话生命周期管理（创建/过期/清理）                    │    │
│  │  2. 阶段路由（opening → optimizing → summary）           │    │
│  │  3. 上下文裁剪与传递（从 Session 提取子 Agent 所需信息）  │    │
│  │  4. 后处理（记忆提取 + 历史压缩）                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐              │
│  │ Diagnosis │      │ Optimize │      │  Report  │              │
│  │  Agent    │      │  Agent   │      │  Agent   │              │
│  │          │      │          │      │          │              │
│  │ 开场诊断  │      │ 简历优化  │      │ 总结报告  │              │
│  │ T=0.3    │      │ T=0.5    │      │ T=0.3    │              │
│  │ 单轮调用  │      │ 短轮调用  │      │ 单轮调用  │              │
│  └──────────┘      └──────────┘      └──────────┘              │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  共享基础设施：                                          │    │
│  │  SessionManager | ConversationMemory | HistoryCompressor │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                     ┌─────────▼─────────┐
                     │  DeepSeek LLM API  │
                     └───────────────────┘
```

### 2.2 文件结构

```
student-value-backend/
├── chat_agent.py          # 协调者 + 会话管理 + 历史压缩 + 记忆
├── multi_agent.py         # 三个专精子 Agent（DiagnosisAgent, OptimizeAgent, ReportAgent）
├── mini_api.py            # API 层（不变）
└── docs/
    └── prd-multi-agent-architecture.md   # 本文档
```

---

## 三、Agent 角色定义

### 3.1 DiagnosisAgent（诊断 Agent）

| 属性 | 说明 |
|---|---|
| **职责** | 基于 HAY 评测结果和简历内容，生成开场诊断 |
| **触发时机** | `start_session()` 时自动调用 |
| **输入** | 评测结果（8 因素 + 5 能力 + 职级 + 薪酬）+ 简历原文 |
| **输出** | 200-300 字的开场诊断文本 |
| **温度** | 0.3（低温度，确保诊断稳定一致） |
| **对话模式** | 单轮（一次输入，一次输出） |

**Prompt 核心内容**：
- HAY 8 因素的完整定义（每个因素的简历信号、常见问题）
- 诊断方法论（亮点 + 短板 + "呈现不足" vs "经历不足"）
- 输出格式规范（简短问候 → 亮点短板 → 优化方向 → 引导用户选择）

**为什么单独拆出来**：
- 诊断需要大量的 HAY 专业知识注入（~2000 字的因素定义），在单 Agent 中会稀释优化和总结的指令
- 诊断只发生一次，用单轮调用保证上下文完全干净
- 低温度保证不同用户的诊断风格一致

### 3.2 OptimizeAgent（优化 Agent）

| 属性 | 说明 |
|---|---|
| **职责** | 根据用户请求改写简历段落，给出 Before/After 对比 |
| **触发时机** | `optimizing` 阶段的每次用户消息 |
| **输入** | 评测结果 + 简历原文 + 对话摘要 + 最近对话 + 结构化记忆 + 用户消息 |
| **输出** | 流式文本（优化建议 + 改写示范 + 维度提升说明） |
| **温度** | 0.5（中温度，平衡创造性和稳定性） |
| **对话模式** | 短轮（携带最近 4 轮对话 + 压缩摘要） |

**Prompt 核心内容**：
- STAR 法则强化（含正反示例对比）
- 量化数据注入技巧
- 动词升级对照表（"负责" → "主导" → "独立负责"）
- 能力维度锚定（每条建议必须关联到具体 HAY 因素）
- 红线约束（不编造经历、不杜撰数据、需要数据时主动追问）

**为什么单独拆出来**：
- 优化需要大量改写示例和技巧（~3000 字），在单 Agent 中和诊断规则互相干扰
- 中温度允许创造性改写，但诊断和总结不需要这种创造性
- 每次调用都传入精准裁剪的上下文（不是全部对话历史），保证行为稳定

### 3.3 ReportAgent（报告 Agent）

| 属性 | 说明 |
|---|---|
| **职责** | 汇总本次对话的所有优化成果，生成总结报告 |
| **触发时机** | `summary` 阶段触发 |
| **输入** | 评测结果 + 对话摘要 + 结构化记忆 + 最近对话 |
| **输出** | 流式文本（优化回顾 + 提升预期 + 后续建议 + 鼓励） |
| **温度** | 0.3（低温度，确保总结准确稳定） |
| **对话模式** | 单轮 |

**Prompt 核心内容**：
- 优化回顾模板（列出每项修改 + 影响维度）
- 提升预期（定性描述，不编造分数）
- 后续建议框架（1-2 个方向）
- 风格要求（温暖专业、正能量收尾、200-300 字）

**为什么单独拆出来**：
- 报告生成依赖结构化记忆（已优化的段落、已达成的共识），不需要全部对话历史
- 低温度保证总结内容准确，不会出现和对话过程矛盾的总结
- 单轮调用，上下文完全干净

### 3.4 ChatAgent（协调者 / Orchestrator）

| 属性 | 说明 |
|---|---|
| **职责** | 会话管理 + 阶段路由 + 上下文传递 + 后处理 |
| **Prompt** | 无（协调者不调用 LLM） |
| **路由方式** | 阶段映射 + 关键词匹配（不使用 LLM 路由，零额外延迟） |

**路由规则**：

```
start_session()
  → DiagnosisAgent.diagnose()
  → 进入 optimizing 阶段

chat() / chat_stream()
  → 检测阶段转换（关键词匹配）
  → 阶段路由：
      optimizing → OptimizeAgent.optimize()
      summary    → ReportAgent.generate_report()
      其他       → OptimizeAgent.optimize()（兜底）
  → 后处理（记忆提取 + 历史压缩）
```

**阶段转换触发词**：`总结`、`结束`、`就这些`、`谢谢`、`没了`、`差不多了`

---

## 四、数据流设计

### 4.1 Session 数据结构

```python
{
    "session_id": "uuid",
    "phase": "opening | optimizing | summary",
    "messages": [{"role": "...", "content": "..."}],   # 完整对话历史
    "assessment_context": {                             # 评测结果
        "factors": {"practical_knowledge": "D+", ...},
        "abilities": {"专业力": {"score": 55, "level": "medium"}, ...},
        "grade": 12,
        "salaryRange": "8.5k~12k",
        "jobTitle": "产品经理",
        "jobFunction": "产品管理"
    },
    "resume_text": "简历原文...",
    "resume_sections": [...],        # 结构化拆分后的段落
    "memory": ConversationMemory(),  # 结构化记忆（跨 Agent 共享）
    "compressed_history": "",        # 压缩后的早期对话摘要
    "message_count": 0               # 总消息计数
}
```

### 4.2 协调者上下文裁剪

协调者从 Session 中提取子 Agent 所需的精准上下文：

```
Session 全量数据
    │
    ├── assessment_context ──────→ 所有子 Agent
    ├── resume_text ─────────────→ 所有子 Agent
    ├── compressed_history ──────→ OptimizeAgent, ReportAgent
    ├── messages[-8:] ───────────→ OptimizeAgent（最近 4 轮原始对话）
    ├── memory.to_context_string()→ OptimizeAgent, ReportAgent
    └── messages[-6:] ───────────→ ReportAgent（最近 3 轮）
```

**设计原则**：子 Agent 只收到它需要的信息，不会被无关上下文干扰。

### 4.3 一次完整对话的数据流

```
Step 1: start_session()
  用户点击"优化简历"
    │
    ├── 创建 Session（存入评测结果 + 简历）
    ├── split_resume_sections()（LLM 拆分简历段落）
    ├── DiagnosisAgent.diagnose()
    │     输入：评测结果 + 简历原文
    │     输出：开场诊断文本
    │     特点：单轮，上下文干净，T=0.3
    ├── 保存到 messages
    └── phase → optimizing

Step 2~N: chat_stream()（优化循环）
  用户发送消息（如"帮我改实习经历"）
    │
    ├── 保存 user message
    ├── 检测阶段转换（关键词匹配）
    ├── _get_agent_context() → 裁剪上下文
    ├── OptimizeAgent.optimize()
    │     输入：评测 + 简历 + 对话摘要 + 最近对话 + 记忆 + 用户消息
    │     输出：流式文本（改写建议 + 前后对比 + 维度提升）
    │     特点：携带最近 4 轮对话保持连贯，T=0.5
    ├── 保存 assistant reply
    └── _post_chat_processing()
          ├── extract_memory()（提取结构化记忆）
          └── compress_history()（如需，压缩早期对话）

Step N+1: chat_stream()（总结）
  用户说"差不多了，帮我总结一下"
    │
    ├── 关键词命中 → phase → summary
    ├── ReportAgent.generate_report()
    │     输入：评测 + 对话摘要 + 记忆 + 最近对话
    │     输出：流式文本（优化回顾 + 提升预期 + 后续建议）
    │     特点：单轮，上下文干净，T=0.3
    └── 保存 assistant reply
```

---

## 五、接口兼容性

### 5.1 对外 API（完全不变）

| 接口 | 方法 | 变化 |
|---|---|---|
| `/api/chat/start` | POST | 不变 |
| `/api/chat/message` | POST | 不变 |
| `/api/chat/history` | GET | 不变 |

### 5.2 内部接口（完全不变）

| 方法 | 签名 | 变化 |
|---|---|---|
| 初始化 | `ChatAgent(client, model)` | 不变 |
| 开始会话 | `start_session(assessment_context, resume_text)` | 不变 |
| 发送消息 | `chat(session_id, user_message)` | 不变 |
| 流式消息 | `chat_stream(session_id, user_message)` | 不变 |
| 获取历史 | `get_history(session_id)` | 不变 |
| 会话管理 | `session_manager.get_session(session_id)` | 不变 |

**零迁移成本**：前端和 API 层完全不需要任何修改。

---

## 六、质量与稳定性分析

### 6.1 质量提升机制

| 机制 | 单 Agent | 多 Agent | 提升原因 |
|---|---|---|---|
| Prompt 详尽度 | 受限（总 prompt 太长会稀释） | 不受限（每个 Agent 独立 prompt） | 每个 Agent 的 prompt 可以写到 5000 字，规则极致详尽 |
| 改写示例数量 | 1-2 个（放多了 prompt 太长） | 10+ 个 | OptimizeAgent 专门放改写示例 |
| HAY 因素定义 | 简略版（节省 token） | 完整版 | DiagnosisAgent 专门放完整定义 |
| 温度调参 | 统一 0.5 | 按场景调：0.3/0.5/0.3 | 诊断和总结用低温度保稳定，优化用中温度保创造性 |

### 6.2 稳定性提升机制

| 机制 | 单 Agent | 多 Agent | 提升原因 |
|---|---|---|---|
| 上下文腐烂 | 聊久了会出现 | 不会出现 | 每个子 Agent 调用都是干净的短上下文 |
| 角色漂移 | 可能忘记当前阶段 | 不可能 | 协调者用规则路由，子 Agent 只有一个角色 |
| 指令遗忘 | prompt 越长越容易 | 不会 | 每个子 Agent 的 prompt 高度聚焦 |
| 自相矛盾 | 可能前后不一致 | 极少 | 结构化记忆 + 对话摘要确保信息一致传递 |

### 6.3 性能影响

| 指标 | 单 Agent | 多 Agent | 说明 |
|---|---|---|---|
| 首次响应延迟 | 无变化 | 无变化 | DiagnosisAgent 单轮调用，与原来相当 |
| 优化阶段延迟 | 无变化 | 无变化 | OptimizeAgent 直接调 LLM，无额外路由 |
| LLM 调用次数/轮 | 1 次对话 + 1 次记忆提取 | 1 次对话 + 1 次记忆提取 | 完全相同，协调者不调 LLM |
| Token 用量 | 较高（全历史 + 大 prompt） | 较低 | 子 Agent 只收到裁剪后的上下文 |

**结论**：多 Agent 架构在延迟和 LLM 调用次数上没有额外开销（因为协调者用规则路由，不调 LLM），甚至在 Token 用量上可能更低（因为上下文裁剪更精准）。

---

## 七、扩展性设计

### 7.1 新增子 Agent 的标准流程

多 Agent 架构的最大优势之一是**即插即用**的扩展性：

```
新增一个 Agent 只需要：
1. 在 multi_agent.py 中新增 Agent 类
2. 定义专精的 System Prompt
3. 实现 run() 方法（同步/流式）
4. 在 ChatAgent 的路由逻辑中添加触发条件
```

### 7.2 未来可扩展的 Agent

| Agent | 职责 | 触发条件 | 依赖 |
|---|---|---|---|
| **SearchAgent** | 搜索招聘网站匹配岗位 | 用户说"帮我找岗位" | Function Call |
| **JDMatchAgent** | 对比简历和 JD，找出差距 | 用户提供 JD 链接 | Function Call |
| **KnowledgeAgent** | 从专业知识库检索 HAY 方法论 | 用户问评测细节 | RAG |
| **MockInterviewAgent** | 模拟面试问答 | 用户说"帮我准备面试" | 新增 Prompt |
| **ValueSimAgent** | 模拟优化后重新评估 | 总结阶段自动触发 | HAY 计算引擎 |

### 7.3 扩展示例：添加 SearchAgent

```python
# multi_agent.py 新增
class SearchAgent:
    SYSTEM_PROMPT = "你是一个岗位搜索专家..."
    TEMPERATURE = 0.3
    TOOLS = [
        {"type": "function", "function": {"name": "search_jobs", ...}}
    ]

    def search(self, query, job_function, city):
        # Function Call 调用搜索 API
        ...

# chat_agent.py 路由新增
if "找岗位" in user_message or "搜索" in user_message:
    return self.search_agent.search(...)
```

---

## 八、风险与应对

| 风险 | 影响 | 应对策略 |
|---|---|---|
| 子 Agent 之间信息丢失 | 优化 Agent 不知道诊断 Agent 说了什么 | 结构化记忆 + 对话摘要确保信息完整传递 |
| 阶段路由不准确 | 用户想结束但关键词没命中 | 可扩展关键词列表；未来可加 LLM 辅助判断 |
| 子 Agent Prompt 迭代 | 改一个 Agent 的 Prompt 可能影响整体体验 | 每个 Agent 独立，改一个不影响其他 |
| 调试复杂度增加 | 出问题需要判断是哪个 Agent | 日志标注 `[DiagnosisAgent]`/`[OptimizeAgent]` 等前缀 |

---

## 九、实现状态

### 9.1 已完成

- [x] DiagnosisAgent 实现（专精 HAY 诊断，T=0.3）
- [x] OptimizeAgent 实现（专精简历改写，T=0.5，支持流式）
- [x] ReportAgent 实现（专精报告生成，T=0.3，支持流式）
- [x] ChatAgent 重构为协调者（阶段路由 + 上下文裁剪）
- [x] API 兼容性验证（零迁移）
- [x] 共享基础设施（SessionManager, ConversationMemory, HistoryCompressor）

### 9.2 待实现

- [ ] SearchAgent（需要 Function Call 支持）
- [ ] JDMatchAgent（需要 Function Call 支持）
- [ ] KnowledgeAgent（需要 RAG 支持）
- [ ] ValueSimAgent（需要 HAY 计算引擎对接）
- [ ] LLM 辅助路由（当关键词匹配不够用时）
- [ ] 子 Agent 调用监控与性能统计
- [ ] A/B 测试框架（对比单 Agent vs 多 Agent 效果）

---

## 十、概念速查表

| 概念 | 一句话解释 | 状态 |
|---|---|---|
| 多 Agent 系统 | 多个专精 Agent 协作，各司其职 | ✅ 已实现 |
| 协调者模式 | 主 Agent 只做路由和管理，不做业务 | ✅ 已实现 |
| 上下文裁剪 | 协调者只传递子 Agent 需要的信息 | ✅ 已实现 |
| 单轮调用 | 子 Agent 每次是全新对话，上下文干净 | ✅ DiagnosisAgent, ReportAgent |
| 短轮调用 | 携带最近几轮对话保持连贯 | ✅ OptimizeAgent |
| 结构化记忆 | 关键信息独立存储，跨 Agent 共享 | ✅ 已实现 |
| 对话压缩 | 早期对话压缩为摘要，控制 token | ✅ 已实现 |
| 按场景调温 | 不同 Agent 使用不同温度 | ✅ 0.3/0.5/0.3 |
| Function Call | LLM 主动调用外部工具 | 🔲 未来 |
| RAG | 先检索知识库再生成 | 🔲 未来 |
