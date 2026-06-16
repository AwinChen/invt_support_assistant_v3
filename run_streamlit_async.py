import asyncio
import pprint
import sys
import os
import re
import traceback
import pytz
from langgraph.checkpoint.mongodb import AsyncMongoDBSaver

from utils.tools import Tools

TIMEZONE = pytz.timezone('Asia/Shanghai')
import torch
# 修复 Streamlit 与 PyTorch 的兼容性问题
torch.classes.__path__ = [os.path.join(torch.__path__[0], "classes")]

from dotenv import load_dotenv
from pymongo import MongoClient

ragflow_path = os.path.join(os.path.dirname(__file__), 'ragflow')
if ragflow_path not in sys.path:
    sys.path.append(ragflow_path)




from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AIMessageChunk
from datetime import datetime, timedelta
import streamlit as st
from graph_asycn import build_async_graph



load_dotenv()

#  (langgraph_app)  streamlit run run_streamlit_async.py
# python -m streamlit run run_streamlit_async.py


st.set_page_config(
    page_title="小英智能客服助手",  # 这个会显示在浏览器的标签页上
    page_icon="👩‍💻",               # 这个会显示在浏览器标签页图标
    layout="centered",           # 页面布局（也可以是 "wide"）
    initial_sidebar_state="auto"
)

def load_text_withtime(text, time, role):
    # 时间戳和内容使用 span，避免额外换行
    if role == "user":
        return f"<span style=\"font-size: 0.9em; color: #666666;\">{time.strftime('%Y-%m-%d %H:%M:%S')}：</span>{text}"
    # 替换 <think>标签为带灰色+小字体样式的 span
    if '<think>' in text:
        text = re.sub(
            r"<think>(.*?)</think>",
            "<span style=\"font-size: 0.9em; color: #666666;\">\\1</div>",
            text,
            flags=re.DOTALL
        )
        # 整体用一个 div 包裹，控制样式，避免额外换行
        html = f"""
            <div style="font-size: 0.9em; color: #666666; line-height: 1.5; margin: 0;">
                <span>{time.strftime('%Y-%m-%d %H:%M:%S')}：</span>{text}
            </div>
            """
    else:
        html = f"<span style=\"font-size: 0.9em; color: #666666;\">{time.strftime('%Y-%m-%d %H:%M:%S')}：</span>{text}"

    return html

def build_langchain_messages(messages):
    return [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in messages
    ]

def save_current_conversation():
    if len(st.session_state.messages):
        st.session_state.conversation_history.append({
            "timestamp": datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.messages.copy(),
            "memory_messages": st.session_state.memory_messages.copy(),
        })
    else:
        pass


def load_conversation(conv_idx):
    # 清空当前的记忆
    st.session_state.messages = []
    st.session_state.memory_messages = []

    selected_conversation = st.session_state.conversation_history[conv_idx]# 加载存储的对话消息到聊天记录
    st.session_state.messages = selected_conversation["messages"]
    st.session_state.memory_messages = selected_conversation["memory_messages"]

    for msg in st.session_state.messages:# 展示历史消息
        with st.chat_message(msg["role"], avatar='☺️' if msg["role"] == 'user' else '👩‍💻'):
            st.markdown(load_text_withtime(msg["content"], msg["time"], msg["role"]), unsafe_allow_html=True)
    st.rerun()


async def get_app():

    MONGO_URL = os.getenv("MONGO_URL")

    checkpointer_cm = AsyncMongoDBSaver.from_conn_string(
        MONGO_URL,
        db_name="invt_SA_checkpoint_db",
        checkpoint_collection_name="checkpoints",
        checkpoint_writes_collection_name="checkpoint_writes"
    )

    checkpointer = await checkpointer_cm.__aenter__()

    app = build_async_graph(checkpointer)

    return app, checkpointer_cm

# ============================================== 界面内容分界线 ==============================================
st.title('👩‍💻 小英智能客服助手 V2.0')
st.markdown("欢迎使用小英智能客服助手，我可以提供GD5000系列型号高压变频器相关问题的咨询服务。")


# 初始化
# 初始化 app（只执行一次）
if "toast_msg" in st.session_state: # 初始化变量，解决 toast_msg 被刷新后不显示问题
    st.toast(st.session_state.toast_msg)
    del st.session_state.toast_msg
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory_messages" not in st.session_state:
    st.session_state.memory_messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []  # 存储所有对话的历史
if "mongodb" not in st.session_state:
    MONGO_URL = os.getenv("MONGO_URL")
    MONGO_DATABASE = os.getenv("MONGO_DATABASE")
    # 初始化mongodb用于日志存储
    mongo_client = MongoClient(MONGO_URL)
    st.session_state.mongodb = mongo_client[MONGO_DATABASE]
if "user_input" not in st.session_state:
    st.session_state.user_input = ''
if "have_response_finished" not in st.session_state:
    st.session_state.have_response_finished = False
if "is_interrupted" not in st.session_state:
    st.session_state.is_interrupted = False



# Sidebar: 历史会话 & 新建对话
with st.sidebar:
    st.header("历史会话")
    st.info("⚠️ 刷新网页后，历史对话将会消失，请知悉。")
    if st.button('+ 保存并新建一个会话'):
        save_current_conversation()
        st.session_state.messages = []
        st.session_state.memory_messages = []
        st.session_state.have_response_finished = False  # 防止重复渲染
        st.session_state.user_input = ''
        st.session_state.toast_msg = "✅ 当前对话已保存"
        # st.toast("✅ 当前对话已保存")

    if len(st.session_state.conversation_history) == 0:
        st.write("暂无对话历史")
    else:
        for idx, conv in enumerate(st.session_state.conversation_history):
            conv_title = f"对话 {idx + 1} - {conv['timestamp']}"
            if st.button(conv_title, key=f"conv_{idx}"):
                load_conversation(idx)
                st.write(f"已加载对话 {idx + 1}: {conv['timestamp']}")

# 主页面按钮: 清除对话记录
if st.button('清除记忆'):
    st.session_state.messages = [] # 清除对话记录按钮
    st.session_state.memory_messages = []  # 清除对话记录按钮
    st.session_state.have_response_finished = False  # 防止重复渲染
    st.session_state.user_input = ''
    st.session_state.toast_msg = "✅ 当前对话已保存"


# 展示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar='☺️' if msg["role"] == 'user' else '👩‍💻'):
        st.markdown(load_text_withtime(msg["content"], msg["time"], msg["role"]), unsafe_allow_html=True)


# 处理新消息
if user_input := st.chat_input("请在此输入您的问题内容"):
    st.session_state.user_input = user_input
    st.session_state.have_response_finished = False

    st.session_state.messages.append({"role": "user", "content": user_input, "time": datetime.now(TIMEZONE)})
    st.session_state.memory_messages.append(HumanMessage(content=user_input))
    with st.chat_message("user", avatar="☺️"):
        st.markdown(load_text_withtime(
            st.session_state.messages[-1]["content"],
            st.session_state.messages[-1]["time"],
            st.session_state.messages[-1]["role"]
        ), unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="👩‍💻"):
        # 使用 st.empty() 创建一个可变区域来模拟动态 spinner
        spinner_placeholder = st.empty() # AI回复框上方的状态提示
        response_placeholder = st.empty() # AI思考过程流式输出框
        stop_button_placeholder = st.empty()

        async def run_async_stream():
            try:
                reply = '' # AI原生回复
                smr_reply = '' # AI摘要回复
                spinner_placeholder.markdown("⏳ <i>正在思考...</i>", unsafe_allow_html=True)
                spinner_placeholder.markdown(
                    f"""
                    <span style="font-size: 0.9em; color: #666666;">
                        ⏳ <i>正在思考...</i>
                    </span>
                    """,
                    unsafe_allow_html=True
                )

                if stop_button_placeholder.button('停止回复'):
                    pass

                state = {
                    "retrieval_docs": '',
                    "total_time": timedelta(seconds=0)
                } # 监听langgraph父图状态机
                app, checkpointer_cm = await get_app()
                app_astream = app.astream(
                        {
                            "user_question": user_input,
                            "messages": build_langchain_messages(st.session_state.messages),
                            "memory_messages": st.session_state.memory_messages.copy(),
                        },
                        config={
                            "recursion_limit": 25,
                            "configurable": {
                                "thread_id": "1"
                            }
                        },
                        stream_mode=["messages", "custom", "values"]
                )

                async for item in app_astream:
                    # print(item)

                    # 兼容多种消息类型
                    if not isinstance(item, tuple) or len(item) < 2:
                        continue
                    msg_type = item[0]


                    if msg_type == "custom":
                        custom_data = item[1]
                        if isinstance(custom_data, dict) and "ai_status" in custom_data:
                            spinner_placeholder.markdown(
                                f"""
                                <span style="font-size: 0.9em; color: #666666;">
                                    ⏳ <i>{custom_data['ai_status']}</i>
                                </span>
                                """,
                                unsafe_allow_html=True
                            )
                        elif isinstance(custom_data, dict) and "fake_stream" in custom_data: # 伪装吐字器
                            reply += custom_data['fake_stream']
                            response_placeholder.markdown(
                                load_text_withtime(reply, datetime.now(TIMEZONE), "assistant"), unsafe_allow_html=True)

                    elif msg_type == "messages":
                        msg_chunk, metadata = item[1]
                        if isinstance(msg_chunk, AIMessageChunk) and metadata.get(
                                "langgraph_node") == "build_prompt_and_invoke":
                            reply += msg_chunk.content
                            response_placeholder.markdown(load_text_withtime(reply, datetime.now(TIMEZONE), "assistant"), unsafe_allow_html=True)

                        elif isinstance(msg_chunk, AIMessageChunk) and metadata.get(
                                "langgraph_node") == "memory_manage":
                            smr_reply += msg_chunk.content

                    elif msg_type == "values":
                        state = item[1]
                        __interrupt__ = state.get("__interrupt__")
                        st.session_state.is_interrupted = True if __interrupt__ else False
                        if st.session_state.is_interrupted:
                            interrupt_obj = __interrupt__[0]
                            interrupt_value = interrupt_obj.value

                            reply = interrupt_value['response_text']
                            response_placeholder.markdown(load_text_withtime(reply, datetime.now(TIMEZONE), "assistant"), unsafe_allow_html=True)


                # 此处为 模型回复完成后，placeholder显示逻辑
                response_placeholder.empty()
                stop_button_placeholder.empty()
                # st.rerun() # 中断回复后刷新界面才能显示UI

                # if state["faq_search_cache"] and len(state["faq_search_cache"]["chunks"])>0:
                #     spinner_placeholder.markdown(
                #         f"""
                #         <span style="font-size: 0.9em; color: #99CC99;">
                #             📚 <i>已深度思考（用时 <b>{state.get("total_time").total_seconds()}</b> 秒）</i>
                #         </span>
                #         """,
                #         unsafe_allow_html=True
                #     )
                if not st.session_state.is_interrupted: # 如果没有中断
                    if state["rag_agent_state"] and len(state["rag_agent_state"]["cache"]["chunks"])>0:
                        chunks = state["rag_agent_state"]["cache"]["chunks"]
                        spinner_placeholder.markdown(
                            f"""
                            <span style="font-size: 0.9em; color: #99CC99;">
                                📚 <i>已深度思考（引用 <b>{len(chunks)}</b> 篇知识库资料，用时 <b>{state.get("total_time").total_seconds()}</b> 秒）</i>
                            </span>
                            """,
                            unsafe_allow_html=True
                        )
                    # elif "data_agent" in state.get("node_status"): # 若走data_agent工作流
                    #     spinner_placeholder.markdown(
                    #         f"""
                    #         <span style="font-size: 0.9em; color: #99CC99;">
                    #             📚 <i>已深度思考（用时 <b>{state.get("total_time").total_seconds()}</b> 秒）</i>
                    #         </span>
                    #         """,
                    #         unsafe_allow_html=True
                    #     )
                    else:
                        spinner_placeholder.markdown(
                            f"""
                            <span style="font-size: 0.9em; color: #993333;">
                                ⚠️ <i>注意：本次回复未引用相关产品知识库,请谨慎辨别内容真实性！(用时<b>{state.get("total_time").total_seconds()}</b>秒)</i>
                        </span>
                        """,
                        unsafe_allow_html=True
                    )

                st.session_state.is_interrupted = False # 重置

                # 为messages变量添加赋值
                st.session_state.messages.append({"role": "assistant", "content": reply, "time": datetime.now(TIMEZONE)})
                st.session_state.memory_messages.append(AIMessage(content=smr_reply))

                st.markdown(load_text_withtime(
                    st.session_state.messages[-1]["content"],
                    st.session_state.messages[-1]["time"],
                    st.session_state.messages[-1]["role"]
                ), unsafe_allow_html=True)


            except Exception as e:
                reply = "⚠️ 对不起，我在处理您的问题时遇到了一点故障,请稍后重试......"
                spinner_placeholder.empty()
                st.session_state.mongodb["qta_log"].insert_one({
                    "raw_question": user_input,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                    "create_time": datetime.now(TIMEZONE)
                })
                print("------------=============== 已捕获本次异常记录 ------------===============")
                # st.error(str(e))
                st.markdown(load_text_withtime(reply, datetime.now(TIMEZONE), "assistant"), unsafe_allow_html=True)

                st.session_state.have_response_finished = False  # 防止重复渲染
                st.session_state.user_input = ''

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(run_async_stream())
        st.session_state.have_response_finished = True

# —— AI 回复结束，下面开始渲染反馈区 ——
# 确保每轮只渲染一次反馈按钮
last_msg = st.session_state.messages[-1] if st.session_state.messages else None
if st.session_state.have_response_finished and last_msg.get("role") == "assistant":
    st.markdown(
        '<span style="font-size:0.9em; font-weight:bold;">本次回答是否对您有帮助：</span>',
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 内容准确", key=f"like_{len(st.session_state.messages)}"):
            # 将反馈存入数据库
            st.session_state.mongodb["feedback_log"].insert_one({
                "raw_question": st.session_state.user_input,
                "response": last_msg.get("content"),
                "feedback": "like",
                "chat_history": Tools.process_chat_messages(build_langchain_messages(st.session_state.messages)),
                "create_time": datetime.now(TIMEZONE)
            })
            st.session_state.toast_msg = "✅ 反馈提交成功！"
            st.session_state.have_response_finished = False  # 防止重复渲染
            st.rerun()  # rerun 后 按钮才会自动消失
    with col2:
        if st.button("👎 内容有误", key=f"dislike_{len(st.session_state.messages)}"):
            st.session_state.mongodb["feedback_log"].insert_one({
                "raw_question": st.session_state.user_input,
                "response": last_msg.get("content"),
                "feedback": "dislike",
                "chat_history": Tools.process_chat_messages(build_langchain_messages(st.session_state.messages)),
                "create_time": datetime.now(TIMEZONE)
            })
            st.session_state.toast_msg = "✅ 反馈提交成功，我们将会改进！"
            st.session_state.have_response_finished = False  # 防止重复渲染
            st.rerun()  # rerun 后 按钮才会自动消失
# ====== 反馈区结束 ======



