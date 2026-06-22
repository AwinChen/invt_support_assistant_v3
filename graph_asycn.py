import json
import operator
import os
import pprint
import re
import sys
import asyncio
import time
from datetime import datetime
import pytz
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ragflow_path = os.path.join(os.path.dirname(__file__), 'ragflow')
if ragflow_path not in sys.path:
    sys.path.append(ragflow_path)

from dotenv import load_dotenv

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.messages import SystemMessage, AIMessageChunk
# from langchain_core.pydantic_v1 import BaseModel, AIMessageChunk
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatTongyi
from langchain_ollama import ChatOllama
# from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, RemoveMessage

from langgraph.graph import END, START, MessageGraph, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.types import Send, Command

from typing import List, Literal, TypedDict, Annotated, Dict, Any, Union

from pymongo import MongoClient

from agent.rag_agent_async import create_rag_agent
from agent.prod_select_agent import create_prod_select_agent

from prompt.prompt_for_graph import RAG_PROMPT, GET_VIDEO_REPLY

from utils.tools import Tools



TIMEZONE = pytz.timezone('Asia/Shanghai')

load_dotenv()
base_url = os.getenv("BASE_URL")
model_name = os.getenv("MODEL_NAME")
api_key = os.getenv("API_KEY")

vbase_url = os.getenv("VBASE_URL")
vmodel_name = os.getenv("VMODEL_NAME")
vapi_key = os.getenv("VAPI_KEY")

MONGO_URL = os.getenv("MONGO_URL")
MONGO_DATABASE = os.getenv("MONGO_DATABASE")

# 初始化mongodb用于日志存储
mongo_client = MongoClient(MONGO_URL)
db = mongo_client[MONGO_DATABASE]
collection = db["qta_log"]


async def fake_stream_output(text, chunk_size=100):
    writer = get_stream_writer()
    content = ''
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        writer({"fake_stream": chunk})
        content += chunk
        await asyncio.sleep(0.02)
    return content


"""
初始化状态机 Graph state
"""
class MainState(TypedDict):
    user_question: str  # 当前对话框用户的提问
    response: str  # 智能客服最终回复

    router_workflow: str  # 当前路由的工作流
    node_status: Annotated[List[str], operator.add]  # 输出运行过程的节点

    # 单词对话计时
    start_time: datetime
    end_time: datetime
    total_time: Any

    # 记忆管理(初始输入带一个HumenMessage)
    memory_messages: List[BaseMessage]  # 经过压缩的多轮对话消息记录,输入AI记忆

    # RAG_agent
    trans_question: str  # 经过LLM处理用以检索的转义问题
    markdown: str
    rag_agent_state: Any

    # prod_select_agent
    prod_select_agent_state: Any  #




async def init_graph(state: MainState, config: RunnableConfig):
    writer = get_stream_writer()
    writer({"ai_status": "正在思考..."})
    # llm = ChatTongyi(model="qwen3.5-27b", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", temperature=0,
    #                  api_key="sk-db286ac12d9940ddb26dd8413279041a", streaming=True)
    # llm_nothink = ChatTongyi(model="qwen3.5-27b", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", temperature=0,
    #                  api_key="sk-db286ac12d9940ddb26dd8413279041a", streaming=True)

    memory_messages = state["memory_messages"]
    user_question = memory_messages[-1].content

    start_time = datetime.now(TIMEZONE)


    return {
        "user_question": user_question,
        "response": '',
        "trans_question": None,
        "memory_messages": memory_messages,
        "start_time": start_time,
        "node_status": [],
        "rag_agent_state": None,
        "prod_select_agent_state": None,
        "data_agent_cache": None,
        "faq_search_cache": None,
    }


async def _compress_messages(input_dict: dict, config: RunnableConfig):
    """在 chain 内对 messages 做压缩，返回压缩后的输入 dict"""
    messages = input_dict["messages"]
    llm_nothink = config['configurable']['llm_nothink']

    # 保留messages第一条系统提示词
    first_system_msg = messages[0] if isinstance(messages[0], SystemMessage) else SystemMessage(content='')

    dialogue_text = Tools.format_messages(messages)

    if len(dialogue_text) >= 10000:
        dialogue_text = dialogue_text[-7500:]

        summary_msg = [
            SystemMessage(content=(
                "你是一个对话摘要器。请基于多轮对话生成一段中文摘要，要求："
                "1. 重点概括用户的核心意图、需求和约束；"
                "2. 重点概括AI已经做过的回复、分析、建议和动作；"
                "3. 如有工具调用、检索、排查、选型、追问等动作，也要概括进去；"
                "4. 保留已确认的关键信息与当前进展；"
                "5. 不要逐轮复述，不要分点，输出一段话即可；"
                "6. 不要引入原对话中没有的新信息；"
                "7. 语言简洁、客观、准确。"
            )),
            HumanMessage(content=f"请总结以下对话：\n\n{dialogue_text}")
        ]
        summary = await llm_nothink.ainvoke(summary_msg, config={"tags": ["_compress_messages"]})
        human_msg = HumanMessage(content=messages[-1].content)

        compressed = [
            first_system_msg,
            SystemMessage(content=f"历史对话摘要：\n{summary.content}"),
            human_msg
        ]
        return {"messages": compressed}

    return input_dict  # 没超限，原样透传


# class Router_struct(BaseModel):
#     reasoning: str = Field(..., description="简要陈述路由依据。核心是分辨用户意图是“产品选型”还是“一般咨询")
#     workflow: Literal["无匹配", "产品选型工作流"] = Field(
#         ...,
#         description=(
#             "路由决策结果，严格按照以下逻辑二选一：\n\n"
#             "- **产品选型工作流**：用户核心意图是“寻找或筛选合适的产品/型号”（如：推荐PLC/变频器产品型号）。只要目的是“找型号”，参数不完整也适用。\n\n"
#             "- **无匹配**：所有非选型的一般咨询场景。（如：具体型号的功能解释、应用场景、接线安装、编程调试、故障排查、工作原理等）。"
#         ),
#     )




async def router(state: MainState, config: RunnableConfig):
    # 　初始化状态机
    memory_messages = state.get("memory_messages", [])
    user_question = memory_messages[-1].content
    if Tools.clean_menu_reply(user_question) in ['我要选型', '产品选型']:
        router_workflow = '产品选型工作流'
    elif Tools.clean_menu_reply(user_question) in ['查看物联网云平台视频', '查看视频']:
        router_workflow = "查看物联网云平台视频"
    else:
        router_workflow = "无匹配"

    return {
        "router_workflow": router_workflow,
        "node_status": ["router"],
    }


"""
定义router条件边
"""
def router_condition(state: MainState):
    if state['router_workflow'] == '产品选型工作流':
        return "prod_select_agent"
    elif state['router_workflow'] == '查看物联网云平台视频':
        return "build_prompt_and_invoke"
    else:
        return "rag_agent"


"""
node: rag_agent
"""
async def rag_agent(state: MainState):  # rag_agent节点

    rag_agent = await create_rag_agent()
    agent_state = {
        "messages": state["memory_messages"],
    }
    rag_agent_state = await rag_agent.ainvoke(agent_state, {"recursion_limit": 15})

    return {
        "node_status": ["rag_agent"],
        "trans_question": rag_agent_state["cache"]["query"],
        "rag_agent_state": rag_agent_state
    }


"""
node: prod_select_agent
"""
async def prod_select_agent(state: MainState):  # rag_agent节点
    writer = get_stream_writer()
    writer({"ai_status": "正在产品选型..."})

    prod_select_agent = await create_prod_select_agent()
    agent_state = {
        "messages": state["memory_messages"],
    }
    prod_select_agent_state = await prod_select_agent.ainvoke(agent_state, {"recursion_limit": 15})

    return {
        "node_status": ["prod_select_agent"],
        "prod_select_agent_state": prod_select_agent_state
    }

"""
node: build_prompt_and_invoke
"""
async def build_prompt_and_invoke(state: MainState, config: RunnableConfig):
    writer = get_stream_writer()
    writer({"ai_status": "正在回复..."})

    # 变量获取
    memory_messages = state.get("memory_messages", [])
    workflow = state["router_workflow"]
    llm = config['configurable']['llm_nothink']


    # 不同工作流的上下文工程
    if workflow == "产品选型工作流":
        cache = state['prod_select_agent_state']["cache"]
        fake_stream_text = cache.get("fake_stream_text", "")

        content = await fake_stream_output(fake_stream_text)
        response = AIMessage(content=content)

    elif workflow == '查看物联网云平台视频':
        content = await fake_stream_output(GET_VIDEO_REPLY)
        response = AIMessage(content=content)

    else:  # RAG工作流，搜知识库
        markdown = state['rag_agent_state']['cache']['markdown']
        rag_prompt = (
            RAG_PROMPT.replace("{datetime_now}", state["start_time"].strftime("%Y年%m月%d日%H时%M分%S秒"))
            .replace("{rag_context}", markdown)
        )

        temp_messages = [SystemMessage(content=rag_prompt)] + memory_messages

        chain = RunnableLambda(_compress_messages) | RunnableLambda(lambda x: x["messages"]) | config["configurable"][
            'llm_nothink']

        response = await chain.ainvoke({"messages": temp_messages}, config={"tags": ["stream"]})

    return {
        "response": response,
        "node_status": ["build_prompt_and_invoke"],
    }


"""
node: memory_manage
"""
async def memory_manage(state: MainState, config: RunnableConfig):
    writer = get_stream_writer()

    memory_messages = state.get("memory_messages", [])
    response = state.get("response")
    llm_nothink = config['configurable']['llm_nothink']
    start_time = state.get("start_time")

    # 记忆记录更新：插入 user + 压缩的ai历史消息（共两条）
    if "<think>" or "</think>" in response.content:  # 清除response中think内容以压缩上下文
        content = re.sub("<think>.*?</think>", '', response.content, flags=re.DOTALL)
        response = AIMessage(content=content)

    # 压缩AI回复
    summary_aimsg = await llm_nothink.ainvoke(
        [HumanMessage(content=f"请用一句话简要概括以下AI回复内容，简洁准确表达核心意思。\nAI回复：{response.content}")])

    # 采用压缩
    memory_messages += [summary_aimsg]

    # 不采用压缩
    # memory_messages += [response]

    end_time = datetime.now(TIMEZONE)
    total_time = end_time - start_time

    # 构建日志以存储智能体输出过程记录
    log = {}
    log["raw_question"] = state.get("user_question", "")
    log["trans_question"] = state.get("trans_question", "")
    log["node_status"] = state.get("node_status")
    log["response"] = state.get("response", "").content if state.get("response", "") else None
    log["router_workflow"] = state.get("router_workflow")
    log["memory_messages"] = str(state.get("memory_messages"))
    log["chat_history"] = Tools.process_data_agent_messages(state.get("memory_messages"))

    # 存储各代理日志
    if state["router_workflow"] == "产品选型工作流":
        log["prod_select_agent_cache"] = json.dumps(
            state["prod_select_agent_state"].get("cache"),
            ensure_ascii=False,
            indent=2
        )

    elif state["router_workflow"] == "无匹配":
        cache = state["rag_agent_state"].get("cache")
        log["rag_agent_markdown"] = cache['markdown']
        log["rag_agent_query"] = cache['query']

    log["total_time"] = total_time.total_seconds()
    log["create_time"] = datetime.now(TIMEZONE)

    await asyncio.to_thread(collection.insert_one, log)

    # 输出关键变量给后端
    writer({"trans_question": state.get("trans_question")})
    writer({"trans_response": summary_aimsg.content})
    chunks_count = (
            (state.get("rag_agent_state") or {})
            .get("cache", {})
            .get("chunks_count")
    )
    writer({"rag_docs_count": chunks_count})
    chunk_md = (
            (state.get("rag_agent_state") or {})
            .get("cache", {})
            .get("markdown")
            or ""
    )
    writer({"chunk_md": chunk_md})

    return {
        "memory_messages": memory_messages,
        "node_status": ["memory_manage"],
        "end_time": end_time,
        "total_time": total_time
    }


"""
构建异步状态图
"""
def build_async_graph(checkpointer=None):
    # 创建异步状态图
    async_invt_ai = StateGraph(MainState)

    async_invt_ai.add_node("init_graph", init_graph)
    # async_invt_ai.add_node("context_compressor", context_compressor)
    async_invt_ai.add_node("router", router)
    async_invt_ai.add_node("rag_agent", rag_agent)
    async_invt_ai.add_node("prod_select_agent", prod_select_agent)
    async_invt_ai.add_node("build_prompt_and_invoke", build_prompt_and_invoke)
    async_invt_ai.add_node("memory_manage", memory_manage)

    async_invt_ai.add_edge(START, "init_graph")
    async_invt_ai.add_edge("init_graph", "router")
    async_invt_ai.add_conditional_edges("router", router_condition, ["rag_agent", "prod_select_agent", "build_prompt_and_invoke"])
    async_invt_ai.add_edge("rag_agent", "build_prompt_and_invoke")
    async_invt_ai.add_edge("prod_select_agent", "build_prompt_and_invoke")
    async_invt_ai.add_edge("build_prompt_and_invoke", "memory_manage")
    async_invt_ai.add_edge("memory_manage", END)

    if checkpointer:
        app = async_invt_ai.compile(checkpointer=checkpointer)
    else:
        app = async_invt_ai.compile()

    return app


if __name__ == '__main__':
    import torch
    from FlagEmbedding import FlagModel
    from ragflow.rag_flow_rerank_model import DefaultRerank
    llm = ChatOpenAI(
        base_url=vbase_url,
        model=vmodel_name,
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
        base_url=vbase_url,
        model=vmodel_name,
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
    embedding_model = FlagModel(os.getenv('EMB_MODEL'),
                                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                use_fp16=torch.cuda.is_available(),
                                empty_init=False,
                                devices="cuda:0"
                                )

    # 异步测试
    async def main():
        app = build_async_graph()
        user_input = input("请输入问题：")
        input_msg = {
            "memory_messages": [HumanMessage(content=user_input)]
        }

        config = {
            "recursion_limit": 25,
            "configurable": {
                "thread_id": 'test_001',
                "llm": llm,
                "llm_nothink": llm_nothink,
                "embedding_model": embedding_model,
                "rerank_model": None,
            }
        }

        while True:
            # response = await app.ainvoke(input_msg, config=config)

            final_state = None
            async for mode, chunk in app.astream(input_msg, config=config,
                                                 stream_mode=["messages", "custom", "values"]):
                # custom stream
                if mode == "custom":
                    print(chunk)

                # token stream
                elif mode == "messages":
                    print(chunk)
                    msg_chunk, metadata = chunk

                    # if msg_chunk.content:
                    # print(msg_chunk.content, end="", flush=True)

                # 最终 state
                elif mode == "values":
                    print(chunk)
                    final_state = chunk

            print("\n")

            next_q = input("👤 你：")

            if next_q.lower() in ["exit", "quit", "退出"]:
                break

            # 必须基于最终 state 继续对话
            input_msg = {
                "messages": final_state.get("messages", []) + [
                    HumanMessage(content=next_q)
                ],
                "memory_messages": final_state.get("memory_messages", []) + [
                    HumanMessage(content=next_q)
                ],
            }


    asyncio.run(main())
