import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver

ragflow_path = os.path.join(os.path.dirname(__file__), 'ragflow')
if ragflow_path not in sys.path:
    sys.path.append(ragflow_path)

import asyncio


from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json
from typing import Optional, List, Dict, Any, AsyncGenerator
from graph_asycn import build_async_graph

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, RemoveMessage, AIMessageChunk
from ragflow.search_result_from_ES_by_question import search_main
import logging
from langgraph.types import Command

import torch
from FlagEmbedding import FlagModel
from ragflow.rag_flow_rerank_model import DefaultRerank
from langchain_openai import ChatOpenAI


load_dotenv()

# 配置日志
logger = logging.getLogger("uvicorn")

# 全局变量
checkpointer = None
langgraph_app = None

llm = None
llm_nothink = None
embedding_model = None
rerank_model = None
global_stop_event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global checkpointer
    global langgraph_app
    global embedding_model
    global rerank_model
    global llm
    global llm_nothink

    MONGO_URL = os.getenv("MONGO_URL")
    EMB_MODEL = os.getenv("EMB_MODEL")
    RERANK_MODEL = os.getenv("RERANK_MODEL")
    VMODEL_NAME = os.getenv("VMODEL_NAME")
    VBASE_URL = os.getenv("VBASE_URL")

    try:
        logger.info("开始加载LLM模型...")
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
        logger.info("LLM模型加载完成")

        # 加载embedding模型
        logger.info("开始加载Embedding模型...")
        embedding_model = FlagModel(EMB_MODEL,
                                    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                    use_fp16=torch.cuda.is_available(),
                                    empty_init=False,
                                    devices="cuda:0"
                                    )
        logger.info(f"Embedding模型加载完成")

        # 加载rerank模型
        # logger.info("开始加载Rerank模型...")
        # rerank_model = DefaultRerank(
        #     key='',
        #     model_name=RERANK_MODEL,
        #     devices="cuda:0"
        # )
        # logger.info(f"Rerank模型加载完成")

        with MongoDBSaver.from_conn_string(
                MONGO_URL,
                db_name="invt_SA_checkpoint_db",
                checkpoint_collection_name="checkpoints",
                checkpoint_writes_collection_name="checkpoint_writes"
        ) as mongo_checkpointer:

            checkpointer = mongo_checkpointer
            langgraph_app = build_async_graph(
                checkpointer=checkpointer
            )
            logger.info("✅ MongoDB检查点存储器和LangGraph应用初始化完成")

            yield

    except Exception as e:
        logger.error(f"❌ 应用启动失败: {e}", exc_info=True)
        raise

    finally:
        logger.info("✅ 资源清理完成")

app = FastAPI(
    title="英威腾智能客服API",
    description="英威腾智能客服系统API",
    version="1.0.0",
    contact={
        "name": "英威腾技术支持",
        "email": "support@invt.com"
    },
    lifespan=lifespan
)



class ChatRequest(BaseModel):
    """定义请求体模型"""
    user_id: str
    session_id: str
    memory_messages: List  # 这是一个由 MessageDict 对象组成的列表


def process_chunk(chunk): # chunk: Tuple
    """
    (
      ("prod_select_agent:08301d1c-3e90-4a76-b34b-d561f42837c9",)
      "messages",
      {
        "AIMessageChunk": {
          "content": "根据",
          "additional_kwargs": {},
          "response_metadata": {},
          "id": "lc_run--019ecf1f-2a54-7270-a50b-fea985e2fbe5"
        },
        "metadata": {
          "thread_id": "002_1",
          "ls_integration": "langchain_chat_model",
          "langgraph_step": 4,
          "langgraph_node": "rag_pipeline",
          "langgraph_triggers": ("branch:to:rag_pipeline",),
          "langgraph_path": ("__pregel_pull", "rag_pipeline"),
          "langgraph_checkpoint_ns": "prod_select_agent:08301d1c-3e90-4a76-b34b-d561f42837c9|rag_pipeline:01f54426-4167-67d5-2cfc-964ab312365b",
          "checkpoint_ns": "prod_select_agent:08301d1c-3e90-4a76-b34b-d561f42837c9",
          "ls_provider": "openai",
          "ls_model_name": "qwen3_14b",
          "ls_temperature": 0.6,
          "ls_max_tokens": 2048
        }
      }
)

    """
    namespace_tuple = chunk[0]
    chunk_type = chunk[1]
    print(chunk)

    # 初始化输出属性
    chunk_response = {
        "chunk_type": chunk_type, # chunk类型
    }

    if chunk_type == "custom":
        custom_data = chunk[2]
        if "ai_status" in custom_data:
            chunk_response["chunk_type"] = "ai_status"
            chunk_response["ai_status"] = custom_data["ai_status"]
            return chunk_response

        elif "fake_stream" in custom_data:
            chunk_response["chunk_type"] = "messages" # 这里对应后端接口字段适配，后端展示AI回复内容格式为 {'chunk_type': 'messages', 'content': '产品选型相'}
            chunk_response["content"] = custom_data["fake_stream"]
            return chunk_response

        elif "trans_question" in custom_data:
            chunk_response["chunk_type"] = "trans_question"
            chunk_response["trans_question"] = custom_data["trans_question"]
            return chunk_response

        elif "trans_response" in custom_data:
            chunk_response["chunk_type"] = "trans_response"
            chunk_response["trans_response"] = custom_data["trans_response"]
            return chunk_response

        elif "rag_docs_count" in custom_data:
            chunk_response["chunk_type"] = "rag_docs_count"
            chunk_response["rag_docs_count"] = custom_data["rag_docs_count"]
            return chunk_response

        elif "chunk_md" in custom_data:
            chunk_response["chunk_type"] = "chunk_md"
            chunk_response["chunk_md"] = custom_data["chunk_md"]
            return chunk_response

    elif chunk_type == "messages":
        # print(chunk)
        msg_chunk, metadata = chunk[2]
        # 流式回复内容
        # if isinstance(msg_chunk, AIMessageChunk) and metadata.get("tags") in ["build_prompt_and_invoke", "rag_pipeline"]:
        if isinstance(msg_chunk, AIMessageChunk) and metadata.get("tags") == ["stream"]:
            chunk_response["content"] = msg_chunk.content
            chunk_response["node"] = metadata.get("langgraph_node")
            return chunk_response


    elif chunk_type == "values":
        state = chunk[2]
        # print(state)
        __interrupt__ = state.get("__interrupt__")
        # print(__interrupt__)
        is_interrupted = True if __interrupt__ else False
        if is_interrupted and namespace_tuple == (): # namespace_tuple == ()表示主图输出，避免子图interrupt重复捕获
            interrupt_obj = __interrupt__[0]
            interrupt_value = interrupt_obj.value

            chunk_response["chunk_type"] = "messages"  # 这里对应后端接口字段适配，后端展示AI回复内容格式为 {'chunk_type': 'messages', 'content': '产品选型相'}
            chunk_response["content"] = interrupt_value['response_text']

            # 返回 【chunk_response】 + 结束标记 【stop_messages】
            return [
                chunk_response,
                {"chunk_type": "stop_messages"} # 后端判断chunk_type==stop_messages时表示结束当前回复的输出
            ]
        else:
            chunk_response["state"] = str(state)
            return chunk_response





# 流式状态转换函数
async def transform_stream(chat_request: ChatRequest, http_request: Request):
    memory_messages = []
    msg_li = chat_request.memory_messages  # 解析请求体
    user_id = chat_request.user_id
    session_id = chat_request.session_id
    thread_id = f"{user_id}_{session_id}"

    # 1. 提前提取最后一条用户消息的内容，作为断点恢复的输入
    latest_user_text = ""
    if msg_li and msg_li[-1]["role"] == "user":
        latest_user_text = msg_li[-1]["content"]

    for msg in msg_li:
        if msg["role"] == "ai":
            memory_messages.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "user":
            memory_messages.append(HumanMessage(content=msg["content"]))

    input_msg = {"memory_messages": memory_messages}

    config = {
        "recursion_limit": 25,
        "configurable": {
            "thread_id": thread_id,
            "llm": llm,
            "llm_nothink": llm_nothink,
            "embedding_model": embedding_model,
            "rerank_model": rerank_model,
        }
    }
    stream_mode = ["messages", "custom", "values"]
    logger.info("Starting LangGraph streaming...")

    try:
        state = await langgraph_app.aget_state(config)
        # print(state)
        is_interrupted = False

        if state.tasks:
            # 检查是否有任何任务携带了未消费的外部中断信号
            is_interrupted = any(task.interrupts for task in state.tasks)

        # 2. 根据检查结果，动态组装 astream 的第一个入参
        if is_interrupted:
            logger.info(f"Thread {config['configurable']['thread_id']} detected interrupt. Resuming with Command.")
            # 唤醒时，直接传入 Command 对象，里面的 resume 会直接喂给节点内的 interrupt() 函数
            astream_input = Command(resume=latest_user_text)
        else:
            logger.info(f"Thread {config['configurable']['thread_id']} is clear. Initiating standard astream.")
            astream_input = input_msg

        # 3. 传入动态决定的参数启动迭代器
        stream_iter = langgraph_app.astream(astream_input, config=config, stream_mode=stream_mode, subgraphs=True)

        while not global_stop_event.is_set():  # 主要检查全局停止信号
            next_task = asyncio.create_task(stream_iter.__anext__()) # ** 关键能够中断原因：只有 cancel asyncio task，LLM 推理才会真正停止，类似你点了外卖，外卖已经在路上，你不要也得送
            try:
                chunk = await asyncio.shield(next_task)
                chunk_response = process_chunk(chunk)

                if chunk_response:
                    if isinstance(chunk_response, list):
                        for item in chunk_response:
                            yield item
                    else:
                        yield chunk_response
                # if chunk_response:
                #     yield chunk_response
            except StopAsyncIteration: # 迭代器自然结束
                logger.info("LangGraph generation has been finished.")
                break
            except asyncio.CancelledError:
                # 这里优先捕获连接中断
                logger.info("transform_stream task was cancelled (likely due to client disconnect.")
                global_stop_event.set()
                break

    except Exception as e:
        logger.error(f"Unexpected error in LangGraph streaming: {e}", exc_info=True)
        yield {"error": "LangGraph processing failed", "detail": str(e)}
    finally:
        global_stop_event.clear()



@app.post("/chat/stream")
async def chat_stream(chat_request: ChatRequest, http_request: Request): # 注入Request
    """流式聊天端点（SSE协议）"""
    async def event_generator():
        # 直接迭代异步生成器
        async for chunk_response in transform_stream(chat_request, http_request): # 将request传递给生成器
            if await http_request.is_disconnected():
                logger.info("Client disconnected detected in event_generator loop.")
                break
            # print(f"data: {json.dumps(chunk_response, ensure_ascii=False)}\n\n")
            yield f"data: {json.dumps(chunk_response, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*"
        }
    )



class SearchRequest(BaseModel):
    """定义请求体模型"""
    question: str

@app.post("/chat/faq_search")
async def faq_search(request: SearchRequest):
    question = request.question
    try:
        result = await asyncio.to_thread(
            search_main,
            question,
            os.getenv("FAQ_TENANT_ID"),
            1,
            0.6
        )
        chunks = result.get("chunks")
        chunk = None
        if len(chunks):
            chunk = chunks[0]["content_with_weight"]
        response_data = {"status": "success", "data": f"{chunk}"}
        return response_data

    except Exception as e:
        error_data = {"status": "errors", "detail": f"An error occurred: {str(e)}"}
        return error_data




if __name__ == "__main__":
    uvicorn.run(
        "run_invt_ai_api:app",  # 修正模块名
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )

"""
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"user_question\": \"变频器报警F51怎么解决\", \"messages\": [], \"memory_messages\": []}"
python run_invt_ai_api.py
python -m uvicorn run_invt_ai_api:app --reload
"""