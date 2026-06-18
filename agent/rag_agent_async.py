import json
import operator
import sys

from dotenv import load_dotenv
import os
import asyncio

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 单元测试用
ragflow_path = 'D:\\python object\\invt_support_assistant\\ragflow'
if ragflow_path not in sys.path:
    sys.path.append(ragflow_path)

from langchain_core.tools import tool, InjectedToolCallId
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AnyMessage

from pydantic import BaseModel, Field
from typing import Annotated, List, Tuple, Union, Literal, Dict, Any
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode, InjectedState
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig


from ragflow.search_result_from_ES_by_question import search_main

from prompt.prompt_for_rag_agent import RAG_SYS_PROMPT

from utils.generate_metadata_to_prompts import Cks_to_Markdown

load_dotenv()
VBASE_URL = os.getenv("VBASE_URL")
VMODEL_NAME = os.getenv("VMODEL_NAME")
VAPI_KEY = os.getenv("VAPI_KEY")

TOPK = int(os.getenv("TOPK"))
SIMILARITY_THRETHOLD = float(os.getenv("SIMILARITY_THRETHOLD"))
CONTEXT_LIMIT = int(os.getenv("CONTEXT_LIMIT"))

RAG_TENANT_ID = os.getenv("RAG_TENANT_ID")
FAQ_TENANT_ID = os.getenv("FAQ_TENANT_ID")
VIDEO_TENANT_ID = os.getenv("VIDEO_TENANT_ID")
IWOSCENE_TENANT_ID = os.getenv("IWOSCENE_TENANT_ID")



# 获取异步流式输出写入器
async def get_async_stream_writer():
    return get_stream_writer()


class Agentstate(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    agent: Any  # 构建的代理
    cache: Annotated[Dict, operator.or_]  # 代理中间缓存


"""
工具参数传入规则定义args_schema：
"""


class rag_search_InputSchema(BaseModel):
    query: str = Field(..., title="有助于高效检索以解答用户需求的检索语句",
                       description="通过上下文对话的意图理解，转换后的适用于向量数据库搜索的高质量中文语言的检索query，该query有助于高效检索中文内容以解答用户需求")
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[Agentstate, InjectedState]


@tool("rag_search", args_schema=rag_search_InputSchema)
async def rag_search(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[Agentstate, InjectedState],
        config: RunnableConfig
) -> Command:
    """
    使用用户转换后的高质量中文检索 query，在向量数据库中进行检索并返回文档列表。
    return: Command
    """
    writer = await get_async_stream_writer()
    writer({"custom_stream": "正在检索知识库..."})

    # 异步执行搜索
    res = await asyncio.to_thread(
        search_main,
        query,
        [RAG_TENANT_ID, IWOSCENE_TENANT_ID, FAQ_TENANT_ID, VIDEO_TENANT_ID, "plc_product_series", "cpq"],
        config['configurable']['embedding_model'],
        config['configurable']['rerank_model'],
        TOPK,
        SIMILARITY_THRETHOLD,
    )

    if res and res.get('chunks'):
        chunks = res.get('chunks')
        markdowns = Cks_to_Markdown().metadata_to_prompts(chunks)
        writer({"custom_stream": f"找到 {len(chunks)} 篇知识库资料..."})
    else:
        chunks = res.get('chunks', [])
        writer({"custom_stream": f"未检索到知识库关联信息..."})
        markdowns = ''

    cache = state["cache"]
    cache["query"] = query
    cache["chunks_count"] = len(chunks)
    # 截断markdown
    if len(markdowns) >= CONTEXT_LIMIT:
        cache["markdown"] = markdowns[:CONTEXT_LIMIT]
    else:
        cache["markdown"] = markdowns

    return Command(
        update={
            "messages": [ToolMessage(content=markdowns, tool_call_id=tool_call_id)],
            "cache": cache
        },
        goto=END
    )


"""
构建图要素
"""
async def init(state: Agentstate, config: RunnableConfig):
    # 创建代理
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYS_PROMPT),
            MessagesPlaceholder(variable_name="messages")
        ],
    )

    return {
        "agent": prompt | config["configurable"]["llm_nothink"].bind_tools(tools=[rag_search], parallel_tool_calls=False),
        "cache": {
            "query": '',
            "chunks_count": '',
            "markdown": ''
        }
    }


async def core_agent(state: Agentstate):
    agent = state["agent"]
    response_msg = await agent.ainvoke({"messages": state["messages"]})

    return {
        "messages": response_msg
    }


async def create_rag_agent():
    subgraph = StateGraph(Agentstate)

    subgraph.add_node("init", init)
    subgraph.add_node("core_agent", core_agent)
    subgraph.add_node("tools", ToolNode(tools=[rag_search]))

    subgraph.add_edge(START, "init")
    subgraph.add_edge("init", "core_agent")
    subgraph.add_conditional_edges("core_agent", tools_condition, {"tools": "tools", END: END})
    subgraph.add_edge("tools", END)

    subgraph_compiled = subgraph.compile()

    return subgraph_compiled


if __name__ == '__main__':
    import torch
    from FlagEmbedding import FlagModel
    from ragflow.rag_flow_rerank_model import DefaultRerank

    llm = ChatOpenAI(
        base_url=VBASE_URL,
        model=VMODEL_NAME,
        api_key='123',
        max_tokens=2048,
        temperature=0.6,
        top_p=0.95,
        presence_penalty=1.5,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0.0,
        },
    )
    llm_nothink = ChatOpenAI(
        base_url=VBASE_URL,
        model=VMODEL_NAME,
        api_key='123',
        max_tokens=2048,
        temperature=0.7,
        top_p=0.80,
        presence_penalty=1.5,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0.0,
        },
    )
    embedding_model = FlagModel(os.getenv("EMB_MODEL"),
                                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                use_fp16=torch.cuda.is_available(),
                                empty_init=False
                                )
    rerank_model = DefaultRerank(
        key='',
        model_name=os.getenv("RERANK_MODEL")
    )
    # llm = ChatTongyi(model="qwen3-8b", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", temperature=0,
    #                  api_key="sk-db286ac12d9940ddb26dd8413279041a", streaming=True)

    input_state = {
        "messages": [
            HumanMessage(content="TS635详细参数")],
    }

    config = {
        "recursion_limit": 25,
        "configurable": {
            "thread_id": '1',
            "llm": llm,
            "llm_nothink": llm_nothink,
            "embedding_model": embedding_model,
            "rerank_model": rerank_model,
        }
    }

    # 异步测试：
    async def main():
        rag_agent = await create_rag_agent()
        response = await rag_agent.ainvoke(
            input_state,
            config=config,
        )
        print(response)


    asyncio.run(main())
