#!/usr/bin/env python3
"""
===========================================
Agent 快速测试脚本（免部署）
===========================================
跳过 Flask / GitHub / Render，直接在本地跑 Agent 验证效果。

用法：
  # 交互式多轮对话（推荐）
  python3 test_agent.py

  # 单条消息快速测试
  python3 test_agent.py "帮我搜索一下上海的产品经理岗位"

  # 跳过开场诊断，直接进优化阶段（更快）
  python3 test_agent.py --fast "帮我改一下实习经历"

  # 只测试工具调用（不走完整 Agent 流程）
  python3 test_agent.py --tool search_jobs '{"keyword":"产品经理","city":"上海"}'
  python3 test_agent.py --tool compare_with_jd '{"jd_text":"岗位要求：熟悉用户增长..."}'
"""

import sys
import os
import json
import time

# 切换到项目目录，确保所有 import 正常
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 加载 .env
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    os.environ.setdefault(key.strip(), value.strip())

load_env()


# ===== 模拟简历和评测数据 =====

MOCK_RESUME = """
教育经历：
- 复旦大学 管理学院 市场营销专业 本科 2022-2026

实习经历：
1. 字节跳动 产品运营实习生 2025.06-2025.09
   - 负责抖音电商某品类的日常运营工作
   - 参与用户增长活动策划，协助完成3场大促活动
   - 整理数据报表，每周汇报运营数据

2. 美团 市场营销实习生 2024.07-2024.09
   - 参与校园推广项目，负责线下物料发放
   - 协助团队进行竞品分析

项目经历：
- 校园二手交易平台 项目负责人 2024.03-2024.06
  开发了一个校内二手物品交易小程序，用户约500人

技能：
- Python, SQL, Excel
- 熟悉常用数据分析工具
"""

MOCK_ASSESSMENT = {
    "jobTitle": "产品经理",
    "jobFunction": "互联网产品",
    "grade": 12,
    "salaryRange": "8k-12k/月",
    "factors": {
        "practical_knowledge": "D",
        "managerial_knowledge": "I",
        "communication": "2",
        "thinking_environment": "D",
        "thinking_challenge": "3",
        "freedom_to_act": "C",
        "magnitude": "N",
        "nature_of_impact": "III",
    },
    "abilities": {
        "专业深度": {"score": 55, "level": "中等"},
        "管理协作": {"score": 35, "level": "较低"},
        "沟通影响": {"score": 45, "level": "中等"},
        "分析决策": {"score": 50, "level": "中等"},
        "业务驱动": {"score": 30, "level": "较低"},
    },
}


def init_services():
    """初始化必要的服务（LLM + 收敛引擎）"""
    from config import config
    from llm_service import LLMService
    from incremental_convergence import IncrementalConvergence
    from validation_rules import validation_rules

    print("初始化服务...")
    llm_service = LLMService(api_key=config.DEEPSEEK_API_KEY, model='deepseek-chat')
    print("  ✓ LLM 服务")

    convergence_engine = IncrementalConvergence(
        validation_rules=validation_rules,
        llm_service=llm_service,
    )
    print("  ✓ 收敛引擎")

    return llm_service, convergence_engine


def init_chat_agent(llm_service, convergence_engine):
    """初始化 ChatAgent"""
    from chat_agent import ChatAgent
    agent = ChatAgent(
        client=llm_service.client,
        model='deepseek-chat',
        llm_service=llm_service,
        convergence_engine=convergence_engine,
    )
    print("  ✓ ChatAgent（含 Function Call）")
    return agent


def test_tool_directly(tool_name, args_json, llm_service, convergence_engine):
    """直接测试单个工具，不走 Agent 流程"""
    from tool_executor import ToolExecutor
    from chat_agent import ConversationMemory

    memory = ConversationMemory()
    memory.set_user_preference("job_title", "产品经理")
    memory.set_user_preference("job_function", "互联网产品")

    executor = ToolExecutor(
        llm_service=llm_service,
        convergence_engine=convergence_engine,
        conversation_memory=memory,
    )
    executor.set_original_assessment({**MOCK_ASSESSMENT, "resume_text": MOCK_RESUME})

    try:
        args = json.loads(args_json)
    except json.JSONDecodeError:
        print(f"❌ 参数 JSON 解析失败: {args_json}")
        return

    print(f"\n🔧 调用工具: {tool_name}({json.dumps(args, ensure_ascii=False)})")
    print("-" * 50)

    start = time.time()
    result = executor.execute(tool_name, args)
    elapsed = time.time() - start

    # 格式化输出
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print(result)

    print(f"\n⏱️  耗时: {elapsed:.1f}s")


def run_single_message(agent, message, fast=False):
    """发送单条消息，打印流式输出"""
    print("\n" + "=" * 50)
    print("开启会话...")
    start = time.time()

    result = agent.start_session(
        assessment_context=MOCK_ASSESSMENT,
        resume_text=MOCK_RESUME,
    )
    session_id = result["session_id"]

    if not fast:
        print(f"\n🤖 开场诊断（{time.time()-start:.1f}s）:")
        print("-" * 50)
        print(result["greeting"])

    print(f"\n👤 用户: {message}")
    print(f"\n🤖 回复:")
    print("-" * 50)

    start2 = time.time()
    for chunk in agent.chat_stream(session_id, message):
        print(chunk, end="", flush=True)
    print(f"\n\n⏱️  回复耗时: {time.time()-start2:.1f}s | 总耗时: {time.time()-start:.1f}s")


def run_interactive(agent):
    """交互式多轮对话"""
    print("\n" + "=" * 50)
    print("开启会话...")
    start = time.time()

    result = agent.start_session(
        assessment_context=MOCK_ASSESSMENT,
        resume_text=MOCK_RESUME,
    )
    session_id = result["session_id"]

    print(f"\n🤖 开场诊断（{time.time()-start:.1f}s）:")
    print("-" * 50)
    print(result["greeting"])
    print()

    print("💡 提示: 输入消息与 Agent 对话，输入 q 退出")
    print("   试试: '帮我搜索一下上海的产品经理岗位'")
    print("   试试: '用优化后的简历重新评估一下'")
    print()

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ('q', 'quit', 'exit'):
            print("👋 再见！")
            break

        print(f"\n🤖 回复:")
        print("-" * 50)

        start2 = time.time()
        for chunk in agent.chat_stream(session_id, user_input):
            print(chunk, end="", flush=True)
        print(f"\n⏱️  耗时: {time.time()-start2:.1f}s\n")


def main():
    args = sys.argv[1:]

    # 解析参数
    fast = "--fast" in args
    if fast:
        args.remove("--fast")

    # 模式 1: --tool 直接测试工具
    if "--tool" in args:
        idx = args.index("--tool")
        if len(args) < idx + 3:
            print("用法: python3 test_agent.py --tool <tool_name> '<json_args>'")
            print("示例: python3 test_agent.py --tool search_jobs '{\"keyword\":\"产品经理\"}'")
            sys.exit(1)
        tool_name = args[idx + 1]
        args_json = args[idx + 2]
        llm_service, convergence_engine = init_services()
        test_tool_directly(tool_name, args_json, llm_service, convergence_engine)
        return

    # 初始化
    llm_service, convergence_engine = init_services()
    agent = init_chat_agent(llm_service, convergence_engine)

    # 模式 2: 单条消息
    if args:
        message = " ".join(args)
        run_single_message(agent, message, fast=fast)
        return

    # 模式 3: 交互式
    run_interactive(agent)


if __name__ == "__main__":
    main()
