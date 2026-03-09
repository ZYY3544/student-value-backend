# Agent 能力升级改造计划

基于 SPARK 评估模型（5.65/10），本文档列出所有改造项的改什么、为什么、怎么改。

---

## P0-1: JD 结构化解析 + 一站式定制改简历

### 改什么
`tool_executor.py` 的 `compare_with_jd` 工具 + 新增 `tailor_resume_to_jd` 工具

### 为什么
当前 `compare_with_jd` 只做匹配度分析（返回百分比+优劣势），但不会：
1. 结构化解析 JD 的底层能力要求（映射到 HAY 体系）
2. 自动基于 JD 差距修改简历（用户需要手动逐段请求修改）

用户愿景是"分析JD → 解析能力要求 → 修改简历一气呵成"。

### 怎么改
1. **升级 `compare_with_jd`**：LLM prompt 增加结构化解析，输出增加 `required_abilities`（JD 要求的能力维度）和 `tailored_suggestions`（针对每个差距的具体改写建议，指向简历的具体段落）
2. **新增 `tailor_resume_to_jd` 工具**：接收 JD 文本，自动完成"解析JD→比对简历→生成定制化改写"的全链路，直接输出改写后的简历段落
3. **在 `multi_agent.py` 的 `OPTIMIZE_AGENT_TOOLS` 中注册新工具**
4. **更新 OptimizeAgent 的 SYSTEM_PROMPT**：增加指令让 Agent 在用户提供 JD 时主动调用一站式链路

### 涉及文件
- `tool_executor.py`：升级 compare_with_jd，新增 tailor_resume_to_jd
- `multi_agent.py`：注册新工具定义 + 更新 prompt

---

## P0-2: Agent 主动执行流

### 改什么
`multi_agent.py` 的 PlanningAgent + OptimizeAgent prompt + `chat_agent.py` 的开场流程

### 为什么
当前 Agent 本质是"你问我答"模式：
- PlanningAgent 只生成简历优化建议，不包含求职阶段规划
- Agent 不会主动搜索岗位、不会主动改简历、不会主动导出 PDF
- 开场只展示 3 个能力选项，没有基于评测结果的个性化引导

用户愿景是 Agent 具备"主动规划→主动搜集→主动改→主动导出"的任务驱动能力。

### 怎么改
1. **升级 PlanningAgent**：
   - 输出结构增加 `proactive_actions`（主动执行建议列表），如 `{"action": "search_jobs", "reason": "你的目标岗位竞争激烈，先看看市场需求", "params": {...}}`
   - 增加宏观求职阶段规划（不只是简历优化）
2. **升级开场流程**（`chat_agent.py` 的 `start_session`）：
   - 利用 PlanningAgent 的结果，在开场白中给出个性化的第一步建议
   - 如果评测结果有明显短板，主动提示"我建议先从XX开始"
3. **升级 OptimizeAgent SYSTEM_PROMPT**：
   - 增加主动任务执行规则：完成改写后主动搜索相关岗位验证效果、完成全部优化后主动提醒导出
   - 增加"求职阶段感知"：根据用户所处阶段（探索期/投递期/面试期）调整交互策略
4. **新增 `generate_pdf` 工具**：Agent 可主动调用生成 PDF

### 涉及文件
- `multi_agent.py`：PlanningAgent prompt + OptimizeAgent prompt + 工具定义
- `chat_agent.py`：start_session 开场逻辑

---

## P1-1: 多版本简历管理

### 改什么
`chat_agent.py` 的 SessionManager + `tool_executor.py` 新增工具 + `multi_agent.py` 更新 prompt

### 为什么
当前系统只维护一份简历文本（`session["resume_text"]`），无法针对不同公司/方向维护多个定制版本。用户愿景是"针对不同投递方向甚至不同公司做定制化简历"。

### 怎么改
1. **Session 新增 `resume_versions` 字段**：`Dict[str, {"label": str, "resume_text": str, "target_jd": str, "created_at": float}]`
2. **新增 `save_resume_version` 工具**：将当前优化后的简历保存为一个命名版本（如"字节跳动-产品经理版"）
3. **新增 `list_resume_versions` 工具**：列出所有已保存的版本
4. **新增 `switch_resume_version` 工具**：切换到指定版本继续优化
5. **更新 OptimizeAgent prompt**：当用户提到不同的投递方向时，主动建议保存当前版本并创建新版本
6. **导出时支持选择版本**

### 涉及文件
- `chat_agent.py`：SessionManager.create_session 增加 resume_versions
- `tool_executor.py`：新增 3 个版本管理工具
- `multi_agent.py`：注册工具 + 更新 prompt

---

## P1-2: 信息真伪识别

### 改什么
`tool_executor.py` 的 `search_jobs` + 新增 `verify_job_posting` 工具

### 为什么
当前搜索结果只有 `source_type` 分类（JD/面经/经验/其他），完全没有真伪判断能力。学生群体是虚假招聘（培训机构伪装、中介骗局、无效挂名岗位）的高发受害群体。

### 怎么改
1. **`search_jobs` 结果增加可信度评估**：
   - 基于 URL 域名打分（zhipin/liepin > 小红书 > 未知来源）
   - 识别培训机构常见关键词（"包就业"、"零基础"、"学费"、"培训"等）
   - 标注可信度等级（高/中/低/疑似虚假）
2. **新增 `verify_job_posting` 工具**：
   - 输入一个 JD 文本或 URL
   - LLM 分析是否存在虚假招聘特征（薪资异常高、要求异常低、公司信息模糊、要求缴费等）
   - 返回可信度评分 + 风险提示
3. **更新 OptimizeAgent prompt**：搜索结果展示时自动标注可信度，遇到可疑信息主动警告用户

### 涉及文件
- `tool_executor.py`：升级 search_jobs + 新增 verify_job_posting
- `multi_agent.py`：注册工具 + 更新 prompt

---

## P1-3: PDF 导出

### 改什么
`resume_export.py` 新增 PDF 生成函数 + `mini_api.py` 新增 API 端点

### 为什么
当前只支持 Word 导出。PDF 是求职简历的标准格式，很多投递系统只接受 PDF。用户愿景中明确提到"主动生成 PDF 版本"。

### 怎么改
1. **`resume_export.py` 新增 `generate_resume_pdf`**：
   - 利用已有的 `generate_resume_docx` 生成 Word
   - 使用 `subprocess` 调用 `libreoffice --headless --convert-to pdf` 进行转换（服务器端通用方案）
   - 如果 libreoffice 不可用，回退到 python-docx2pdf 或直接用 reportlab 生成
2. **`mini_api.py` 新增 `/api/chat/resume/export-pdf` 端点**
3. **注册 `export_pdf` 工具**，让 Agent 可以主动触发 PDF 生成并返回下载链接

### 涉及文件
- `resume_export.py`：新增 generate_resume_pdf
- `mini_api.py`：新增 API 端点
- `tool_executor.py`：新增 export_pdf 工具
- `multi_agent.py`：注册工具

---

## P2-1: 职业认知框架（AI 时代求职引导）

### 改什么
`multi_agent.py` 的 OptimizeAgent + PlanningAgent prompt

### 为什么
当前 Agent 只帮学生解决 how（怎么改简历），但不帮学生理解 why（为什么选这个方向）和 what（市场需要什么）。在 AI 时代，很多传统岗位正在被重塑，学生需要认知引导。

### 怎么改
1. **OptimizeAgent SYSTEM_PROMPT 增加"认知引导"模块**：
   - 当用户表达迷茫（"不知道该找什么工作"、"不确定方向"）时，触发引导式提问
   - 引导框架：兴趣→能力→市场需求 三圈交集
   - AI 时代岗位趋势认知：哪些岗位在增长、哪些在萎缩、AI 如何改变各行业
2. **PlanningAgent 增加"认知诊断"维度**：
   - 判断用户是否处于"迷茫期"（目标模糊、投递方向不明确）
   - 如果是，优化计划中增加认知引导建议
3. **新增"行业趋势"搜索关键词模板**：当用户询问行业/方向时，search_jobs 自动附加行业趋势搜索

### 涉及文件
- `multi_agent.py`：OptimizeAgent + PlanningAgent prompt 大幅更新

---

## P2-2: 多 JD 横向对比

### 改什么
`tool_executor.py` 新增 `compare_multiple_jds` 工具

### 为什么
学生经常同时考虑多个岗位方向，需要知道"哪个岗位最适合我"。当前只能一次比一个 JD，没有横向对比能力。

### 怎么改
1. **新增 `compare_multiple_jds` 工具**：
   - 接收 2-4 个 JD 文本
   - 分别计算匹配度
   - 横向对比各 JD 的能力要求差异
   - 输出推荐排序 + 每个 JD 的投递优先级建议
2. **更新 OptimizeAgent prompt**：当用户说"帮我看看这几个岗位哪个更适合"时，引导使用多 JD 对比

### 涉及文件
- `tool_executor.py`：新增 compare_multiple_jds
- `multi_agent.py`：注册工具 + 更新 prompt

---

## 改造顺序与依赖关系

```
P0-1 (JD一站式) ──┐
                   ├── P1-1 (多版本管理，依赖定制化改写能力)
P0-2 (主动执行流) ─┤
                   ├── P1-3 (PDF导出，主动执行流需要调用)
                   │
P1-2 (真伪识别)    │  独立，可并行
                   │
P2-1 (认知框架) ───┤  依赖主动执行流的引导能力
P2-2 (多JD对比) ───┘  依赖 JD 解析升级
```

## 已完成（代码已提交）

| 改造项 | 状态 | 说明 |
|--------|------|------|
| P0-1 JD 结构化解析 + 一站式定制 | ✅ 已完成 | tailor_resume_to_jd + compare_with_jd 升级 |
| P0-2 Agent 主动执行流 | ✅ 已完成 | PlanningAgent 升级 + 个性化开场 + 代码级自动触发 JD 建议 |
| P1-1 多版本简历管理 | ✅ 已完成 | save/list/switch + Supabase 持久化 + 跨会话加载 |
| P1-2 信息真伪识别 | ✅ 已完成 | 规则引擎 + LLM 深度分析 + 搜索结果自动标注 |
| P1-3 PDF 导出 | ✅ 已完成 | libreoffice → fpdf2 双方案 + API 端点 |
| P2-1 职业认知框架 | ✅ 已完成 | 迷茫期引导 + AI 时代趋势 + 能力画像解读 |
| P2-2 多 JD 横向对比 | ✅ 已完成 | compare_multiple_jds 工具 |

## 需要你在电脑端完成的事项

### 1. Supabase 建表（必须）
简历版本持久化需要在 Supabase 中创建 `resume_versions` 表：
```sql
CREATE TABLE resume_versions (
    id TEXT PRIMARY KEY,           -- 格式: {session_id}_{version_id}
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    version_id TEXT NOT NULL,      -- 如 v_1, v_2
    label TEXT NOT NULL,           -- 如「字节跳动-产品经理版」
    resume_text TEXT NOT NULL,
    target_jd TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_resume_versions_user ON resume_versions(user_id);
```

### 2. PDF 中文字体安装（建议）
服务器部署时需要中文字体支持。两种方案：

**方案 A：系统安装字体（推荐）**
```bash
# Ubuntu/Debian
apt-get install fonts-wqy-microhei

# CentOS/RHEL
yum install wqy-microhei-fonts
```

**方案 B：项目目录放字体文件**
```bash
mkdir -p fonts/
# 下载 Noto Sans SC 字体到 fonts/ 目录
wget -O fonts/NotoSansSC-Regular.ttf \
  "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC-Regular.ttf"
```

### 3. 安装 fpdf2 依赖（必须）
```bash
pip install fpdf2
# 或在 requirements.txt 中添加
echo "fpdf2>=2.7.0" >> requirements.txt
```

### 4. Docker 镜像更新（如果使用 Docker 部署）
在 Dockerfile 中添加：
```dockerfile
RUN apt-get update && apt-get install -y \
    libreoffice-writer \
    fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*
```

### 5. 接入天眼查/企查查 API（可选，冲 9 分）
当前真伪识别基于关键词规则 + LLM 分析，没有实时工商信息验证。
如果要增强可信度：
- 注册天眼查开放平台 API（有免费额度）
- 在 `verify_job_posting` 中增加公司名工商查询
- 验证公司是否存在、注册资本、经营状态等

### 6. 行业趋势数据源（可选，冲 9 分）
当前职业认知引导依赖 LLM 通用知识，没有实时数据。
可考虑：
- 定期爬取招聘平台的行业报告/趋势数据
- 接入脉脉或猎聘的行业数据 API
- 在项目中维护一个 `career_knowledge.json` 文件，定期更新行业趋势

## 实际评分

经过两轮改造，SPARK 评分从 5.65 提升至 7.81：
- S (智能简历): 6.9 → 8.25
- P (主动规划): 5.8 → 7.83
- A (分析匹配): 4.8 → 8.17
- R (信息搜集): 4.5 → 7.0
- K (认知引导): 5.2 → 7.17

完成上述电脑端配置后，预计可达 8.5+。
