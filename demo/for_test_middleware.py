import os

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
    ModelFallbackMiddleware,
    ToolCallLimitMiddleware,
    PIIMiddleware,
    LLMToolSelectorMiddleware,
    TodoListMiddleware
)

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi





@tool
def query_alarm(code: str) -> str:
    """查询变频器故障码"""
    db = {
        "E101": "过流故障",
        "E102": "过压故障",
        "E103": "欠压故障"
    }
    return db.get(code, "未知故障")


@tool
def get_solution(code: str) -> str:
    """查询处理建议"""
    solutions = {
        "E101": "检查电机和负载",
        "E102": "检查输入电压",
        "E103": "检查电源线路"
    }
    return solutions.get(code, "暂无建议")


llm = ChatOpenAI(
        base_url="http://192.168.5.197:8000/v1",
        model="qwen3_14b",
        api_key="czy0109248",
        max_tokens=6000,
        temperature=0.6,
        top_p=0.95,
        presence_penalty=1.0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0.0,
        },
    )
# llm = ChatTongyi(model="qwen-plus", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", temperature=0,
#                      api_key="sk-db286ac12d9940ddb26dd8413279041a", streaming=False)

agent = create_agent(
    model=llm,
    tools=[query_alarm, get_solution],
    middleware=[TodoListMiddleware()],
)
response = agent.invoke({"messages":"先分析变频器出现 E103 原因，再查处理建议，再整理成排查步骤"})
# response = llm.invoke("变频器出现E103该如何解决？")

print(response)