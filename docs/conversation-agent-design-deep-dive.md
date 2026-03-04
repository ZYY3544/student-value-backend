# Agent 设计深度探讨：Function Call、RAG、MCP 与长上下文优化

> 日期：2026-03-04
> 背景：基于已完成的简历优化 Agent，围绕架构演进方向和长上下文最佳实践展开的深度探讨。

---

## 目录

1. [Function Call 与 RAG 概念辨析](#1-function-call-与-rag-概念辨析)
2. [产品演进思路：联网、知识库与 MCP](#2-产品演进思路联网知识库与-mcp)
3. [长上下文腐烂问题与解决策略](#3-长上下文腐烂问题与解决策略)
4. [长上下文 Prompt 最佳实践：位置优先与降噪排序](#4-长上下文-prompt-最佳实践位置优先与降噪排序)
5. [最终实现：Prompt 重构落地](#5-最终实现prompt-重构落地)

---

## 1. Function Call 与 RAG 概念辨析

### 问题

> 在我们这次 Agent 的设计中，Function Call 和 RAG 是否都被用到了？

### 结论：两个都没有用到

#### Function Call（函数调用/工具调用）

**是什么**：LLM 在对话过程中，自己判断"我需要调用一个外部工具"，然后输出一个结构化的函数调用请求，系统执行后把结果喂回给 LLM，LLM 再继续生成。

示例流程：

```
用户: "帮我查一下北京今天天气"
LLM → 决定调用 get_weather(city="北京")
系统 → 执行函数，返回 "晴，25°C"
LLM → "北京今天晴天，25度，适合出门"
```

**我们用了吗**：没有。我们的 Agent 全程只做文本对话，LLM 不会主动调用任何外部工具。所有的信息（简历、评测结果）都是在 prompt 里直接给它的。

#### RAG（Retrieval-Augmented Generation，检索增强生成）

**是什么**：先把大量文档切片存到向量数据库，用户提问时，先检索出最相关的几段内容，塞进 prompt 里，再让 LLM 回答。核心是**先检索、再生成**。

示例流程：

```
用户: "HAY体系中PK因素怎么提升？"
系统 → 去向量库搜索，找到3段相关文档
系统 → 把这3段塞进prompt
LLM → 基于检索到的内容回答
```

**我们用了吗**：没有。我们是把简历全文和评测结果**一股脑全塞进 prompt**，没有检索这一步。这叫**上下文注入（Context Injection）**，不是 RAG。

#### 我们实际用了什么

| 技术 | 说明 |
|---|---|
| 动态 Prompt 工程 | 根据对话阶段（开场→优化→总结）切换不同的 system prompt |
| 上下文注入 | 把简历 + HAY 评测结果直接写进 prompt |
| 会话管理 | 内存维护多轮对话历史 |
| SSE 流式输出 | 实时逐字返回 |

#### 什么时候会用到它们？

- **Function Call**：如果想让 Agent 能**主动去做事**，比如"帮我改完简历后自动重新跑一次评测"、"去招聘网站搜索匹配岗位"，那就需要给 LLM 定义工具让它自己调用。
- **RAG**：如果想让 Agent 掌握一个**大型知识库**（比如几百页的 HAY 评估手册、各行业薪酬报告），简历太长塞不进 prompt，那就需要 RAG 来按需检索相关段落。

简单说，我们目前的 Agent 是一个**"上下文感知的多轮对话机器人"**，还不算严格意义上的"有工具能力的 Agent"。加上 Function Call 和 RAG 是下一步进化的方向。

---

## 2. 产品演进思路：联网、知识库与 MCP

### 用户的产品设想

> 1. 用户能在这里去招聘网站上搜索匹配岗位，然后对着岗位再修改简历
> 2. 用户能查询到意愿公司在官网上发布的最新岗位，然后直接比较
> 3. 上传很多优质简历和岗位价值评估方法论，让信息来源更干净

### 联网 = Function Call

**联网搜索就是 Function Call 的典型应用场景。**

流程大概是这样的：

```
用户: "帮我看看字节跳动最近有什么产品经理岗位"

LLM → 判断需要调用工具 → search_jobs(company="字节跳动", role="产品经理")
系统 → 执行爬虫/调API → 返回岗位列表
LLM → "我找到了3个匹配岗位，我们来逐个对比你的简历..."
```

LLM 自己不能上网，但通过 Function Call，它可以**指挥系统去做事**，然后拿到结果继续对话。具体场景：

- 搜招聘网站 → `search_jobs()` 工具
- 查公司官网岗位 → `fetch_company_jobs()` 工具
- 对比简历和 JD → LLM 拿到岗位信息后直接在对话里分析

这些都是给 LLM 定义好工具，它自己判断什么时候该调用。

### RAG 让信息源更干净

RAG 的核心价值就是让模型的信息来源**更精准、更专业**，而不是靠通用训练知识。

```
没有 RAG：
  LLM 靠自己的通用知识回答 → 信息太 general，可能不准

有 RAG：
  上传 → 100份优质简历 + HAY方法论文档 + 行业薪酬报告
  系统 → 切片、向量化、存入向量数据库
  用户提问 → 先检索最相关的几段 → 塞进prompt
  LLM → 基于专业资料回答 → 信息源干净、可控
```

对比示例：

| 场景 | 没有 RAG | 有 RAG |
|---|---|---|
| "帮我优化财务总监简历" | 靠通用知识给建议 | 检索出库里3份优质财务总监简历作为参考 |
| "HAY体系里技术专家该怎么评？" | 可能瞎编 | 从上传的HAY方法论文档中检索到精确定义 |

所以 RAG 不是让模型"更聪明"，而是让它**有据可依**，减少幻觉。

### MCP（Model Context Protocol）

MCP = **模型上下文协议**，是 Anthropic 提出的一个**开放标准**。

简单理解：**MCP 是 Function Call 的"标准化升级版"。**

```
Function Call 的问题：
  每个 LLM 厂商的工具调用格式都不一样
  给 DeepSeek 写的工具，换到 GPT 就得改
  每接一个外部服务，都要自己写适配代码

MCP 解决的问题：
  定义了一套统一的协议标准
  外部服务只要实现 MCP Server，任何支持 MCP 的 LLM 都能直接用
  类似于 USB 接口 —— 不管什么设备，插上就能用
```

对比：

```
没有 MCP：
  Agent想调招聘网站 → 自己写代码对接
  Agent想调简历解析 → 自己写代码对接
  Agent想调薪酬数据库 → 自己写代码对接
  （每个都要从头搞）

有 MCP：
  招聘网站提供 MCP Server → Agent直接连
  简历解析提供 MCP Server → Agent直接连
  薪酬数据库提供 MCP Server → Agent直接连
  （即插即用）
```

**我们这个项目用到 MCP 了吗？没有。** 连 Function Call 都还没用，MCP 是更上层的东西。

### 产品进化路线

```
当前（已完成）
  ✅ 多轮对话 + 上下文注入 + 流式输出

第二步
  🔲 Function Call → 联网搜岗位、查公司官网

第三步
  🔲 RAG → 导入优质简历库 + HAY方法论文档，让建议有据可依

第四步
  🔲 MCP → 标准化接入各种外部服务（招聘平台、薪酬数据库等）
```

---

## 3. 长上下文腐烂问题与解决策略

### 问题

> 我们这个 Agent 可能出现非常长的上下文，会不会出现上下文腐烂？有什么解决策略？

### 结论：一定会出现

当对话越来越长，LLM 会出现：

| 症状 | 表现 |
|---|---|
| 遗忘 | 聊到后面，忘了前面简历里写的关键信息 |
| 迷失重点 | 被中间大量对话淹没，抓不住核心要点 |
| 自相矛盾 | 前面说"你适合管理岗"，后面又说"建议走技术路线" |
| 指令漂移 | 忘了系统 prompt 里的阶段规则，行为变得随意 |

我们的场景特别容易中招，因为一次完整对话可能包含：系统 prompt（很长）+ 简历全文 + 评测结果 + 十几轮修改讨论，轻松超过几万 token。

### 策略一：对话摘要压缩（最实用）

每隔 N 轮，让 LLM 自己总结之前的对话，用摘要替换原始历史。

```python
def maybe_compress(messages, threshold=20):
    if len(messages) < threshold:
        return messages

    # 保留系统 prompt（永远不丢）
    system = messages[0]

    # 让 LLM 把前面的对话压缩成摘要
    summary = llm_summarize(messages[1:-6])  # 保留最近6轮不压缩

    # 重组：系统prompt + 摘要 + 最近几轮原始对话
    return [system, {"role": "assistant", "content": summary}] + messages[-6:]
```

**效果**：token 用量大幅下降，最近的对话保持精确，历史信息保留要点。

### 策略二：关键信息锚定

把重要信息**反复注入**，不让它被淹没。

```python
def build_system_prompt(phase, resume, assessment, agreed_changes):
    return f"""
{phase_prompt}

【简历核心信息 - 每轮必读】
{resume_summary}

【评测结论 - 每轮必读】
{assessment_summary}

【本次对话已达成的共识】
{agreed_changes}  # 动态更新
"""
```

**核心思想**：重要的东西不要只说一次然后指望 LLM 记住，而是每轮都重复强调。

### 策略三：分段式会话（Session Splitting）

一次大对话拆成多个独立小会话，每段有明确目标。

```
Session 1: 简历诊断 → 输出诊断报告（存下来）
Session 2: 带着诊断报告 → 逐项优化（存中间结果）
Session 3: 带着优化结果 → 最终润色和总结
```

每个 session 都是**干净的短上下文**，通过中间结果传递信息，而不是靠一个超长对话硬撑。

### 策略四：结构化记忆存储

把关键信息从对话流中抽出来，存到结构化的地方。

```python
class ConversationMemory:
    def __init__(self):
        self.resume_summary = ""      # 简历摘要
        self.assessment_result = {}    # 评测结论
        self.agreed_modifications = [] # 已同意的修改项
        self.user_preferences = {}     # 用户偏好（如意向岗位）

    def update(self, key, value):
        """每轮对话后，提取关键信息存储"""
        ...

    def inject(self, system_prompt):
        """每轮调用前，把记忆注入 prompt"""
        return system_prompt + self.to_context_string()
```

**对话历史可以丢，但结构化记忆不会丢。**

### 推荐的组合方案

```
                    ┌─────────────────┐
                    │   System Prompt  │ ← 每轮都带，永远不丢
                    ├─────────────────┤
                    │  结构化记忆       │ ← 关键信息锚定，每轮注入
                    │  - 简历摘要       │
                    │  - 评测结论       │
                    │  - 已达成共识     │
                    ├─────────────────┤
                    │  历史对话摘要     │ ← 早期对话压缩成摘要
                    ├─────────────────┤
                    │  最近 4-6 轮原始  │ ← 保持最近对话的精确性
                    │  对话记录         │
                    └─────────────────┘
```

---

## 4. 长上下文 Prompt 最佳实践：位置优先与降噪排序

### 来源

来自长上下文 Prompt 最佳实践总结。

### 策略一：位置优先（首尾强化）

LLM 的注意力分布是 **U 形曲线** —— 对开头和结尾的内容关注度最高，中间容易被忽略。

**原则**：将最关键的系统指令和 few-shot 示例放在 prompt 的"首部"，利用最高注意力权重建立任务基调；同时在"尾部"也加上核心提醒进行强化。例如使用 Function Call 时，把 Tool 定义放在 messages 的末尾可以提升 Function Call 的成功率。

对应到我们项目：

```
原来（扁平结构）：
  System Prompt = 角色 + 简历 + 评测 + 阶段规则 + ... 全混在一起

优化后（首尾强化）：
  System Prompt 开头 = 身份认知 + 核心行为规则（最重要的）
  System Prompt 结尾 = "⚠️ 核心提醒：你当前处于XX阶段，必须..."（强化）
```

### 策略二：降噪排序（相关性过滤）

对冗长的历史对话和相关性参差不齐的 RAG 文档，先做预处理，去除不相关的噪音，再按照与当前任务的相关性排序，相关性更高的内容（如历史对话）放在前面。

对应到我们项目：

```
用户聊了20轮，其中：
  - 3轮在讨论工作经历的量化数据 ← 高相关
  - 5轮在闲聊、寒暄、问无关问题  ← 低相关，可以丢掉
  - 2轮在确认格式偏好            ← 中等相关，压缩保留

降噪后只保留高相关内容，噪音不再干扰 LLM 判断
```

### 结合后的 Prompt 模板结构

```
System Prompt:
  # 核心身份 + 行为规则（首部，高注意力）

User Prompt:
  <history>
    # 经过降噪、压缩后的对话历史
    # 按相关性排序，重要的靠前
  </history>

  <context>
    # 简历内容 + 评测结果
    # 未来加 RAG 后，检索到的文档也放这里
  </context>

  <task>
    # 当前阶段的精简指令（尾部，高注意力）
    # "你现在处于【优化阶段】，请针对用户刚才的回答..."
  </task>
```

**关键变化**：当前阶段的核心指令放在 `<task>` 里，也就是整个 prompt 的**最末尾**，利用尾部的高注意力权重来防止阶段漂移。

### 四大策略的组合

| 策略 | 作用 |
|---|---|
| 位置优先 | 首尾放核心指令，防止指令漂移 |
| 降噪排序 | 过滤无关对话，减少噪音 |
| 对话摘要压缩 | 早期对话压缩成摘要，控制 token |
| 结构化记忆 | 关键信息抽出来单独存，每轮注入 `<context>` |

---

## 5. 最终实现：Prompt 重构落地

### 改动总览（+401 行，-58 行）

基于以上讨论，对 `chat_agent.py` 进行了完整重构。

### 5.1 新增 `ConversationMemory` 结构化记忆

```python
class ConversationMemory:
    def __init__(self):
        self.resume_summary: str = ""
        self.assessment_highlights: str = ""
        self.assessment_weaknesses: str = ""
        self.agreed_modifications: List[str] = []  # 已达成的修改共识
        self.user_preferences: Dict[str, str] = {} # 用户偏好
        self.optimized_sections: List[str] = []    # 已优化过的段落
        self.pending_questions: List[str] = []     # 待用户补充的信息
```

- 独立存储关键信息，不随对话历史压缩而丢失
- 每轮对话后自动用 LLM 提取关键信息存入
- 通过 `to_context_string()` 方法转为可注入 prompt 的文本

### 5.2 新增 `HistoryCompressor` 对话压缩与降噪

```python
class HistoryCompressor:
    COMPRESS_THRESHOLD = 16  # 超过8轮触发压缩
    KEEP_RECENT = 8          # 最近4轮原始保留

    @staticmethod
    def compress_history(client, model, session) -> str:
        """用 LLM 将早期对话压缩为200-400字摘要"""

    @staticmethod
    def extract_memory(client, model, session):
        """从最近一轮对话中提取结构化记忆"""
```

- 超过 8 轮（16 条消息）自动触发 LLM 摘要压缩
- 最近 4 轮原始对话始终保留，保持精确性
- 压缩后裁剪消息列表，控制 token 用量

### 5.3 重构 `PromptBuilder` 为首尾强化结构

```
┌──────────────────────────────────┐
│ System Prompt（首部·高注意力）     │  ← 核心身份 + 行为规则
├──────────────────────────────────┤
│ 压缩摘要 + 最近4轮原始对话        │  ← 降噪后的历史
├──────────────────────────────────┤
│ <context> 评测+简历+结构化记忆    │  ← 关键信息锚定
├──────────────────────────────────┤
│ <task> 阶段指令 + ⚠️核心提醒     │  ← 尾部·高注意力，防阶段漂移
└──────────────────────────────────┘
```

关键设计：

- `build_system_prompt()` 只放核心身份和行为规则（首部高注意力）
- `build_user_prompt()` 构建 `<history>` + `<context>` + `<task>` 三段式结构
- 每个阶段指令末尾增加 `⚠️ 核心提醒`，利用尾部注意力防止阶段漂移

### 5.4 新增 `_post_chat_processing()` 每轮后处理

```python
def _post_chat_processing(self, session_id):
    # 1. 提取结构化记忆
    HistoryCompressor.extract_memory(self.client, self.model, session)
    # 2. 判断是否需要压缩
    if HistoryCompressor.should_compress(session):
        compressed = HistoryCompressor.compress_history(...)
        # 更新摘要，裁剪消息列表
```

在 `chat()` 和 `chat_stream()` 中都调用，确保每轮对话后自动维护记忆和压缩。

---

## 附：关键概念速查表

| 概念 | 一句话解释 | 我们用了吗 |
|---|---|---|
| Function Call | LLM 自己决定调用外部工具 | ❌ 未用 |
| RAG | 先检索相关文档，再塞进 prompt 生成 | ❌ 未用 |
| MCP | Function Call 的标准化协议（即插即用） | ❌ 未用 |
| Context Injection | 把信息直接塞进 prompt | ✅ 在用 |
| 上下文腐烂 | 长对话导致 LLM 遗忘/矛盾/漂移 | ✅ 已解决 |
| 位置优先 | 利用首尾高注意力放关键指令 | ✅ 已实现 |
| 降噪排序 | 过滤无关对话，按相关性排序 | ✅ 已实现 |
| 结构化记忆 | 关键信息独立存储，不随压缩丢失 | ✅ 已实现 |
| 对话摘要压缩 | LLM 将早期对话压缩为摘要 | ✅ 已实现 |
