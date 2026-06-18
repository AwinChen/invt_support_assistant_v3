import streamlit as st
import requests
import json
import uuid

# python -m streamlit run streamlit_test_app.py

# 设置页面基本配置
st.set_page_config(page_title="小英智能客服 - 测试终端", page_icon="🤖", layout="centered")
st.title("🤖 小英智能客服api测试")

# 初始化 Session State (保持会话记忆和ID)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = "test_user_001"
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏配置面板
with st.sidebar:
    st.header("⚙️ 测试配置")
    api_url = st.text_input("API 地址", value="http://127.0.0.1:8000/chat/stream")

    st.session_state.user_id = st.text_input("User ID", value=st.session_state.user_id)
    st.session_state.session_id = st.text_input("Session ID", value=st.session_state.session_id)

    if st.button("🗑️ 清空历史记录"):
        st.session_state.messages = []
        st.rerun()

# 渲染历史聊天记录
for msg in st.session_state.messages:
    # 注意：Streamlit 默认识别 "user" 和 "assistant"图标，我们把你的 "ai" 映射一下
    role = "assistant" if msg["role"] == "ai" else msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])

# 聊天输入框
if prompt := st.chat_input("请输入您的问题 (例如：产品选型相关问题)..."):

    # 1. 立即把用户消息追加到界面和状态中
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 准备接收 AI 回复
    with st.chat_message("assistant"):
        # 占位符：用于展示流式文本和状态提示
        status_placeholder = st.empty()
        message_placeholder = st.empty()

        full_response = ""

        # 构造请求体 (与你的 ChatRequest BaseModel 对应)
        payload = {
            "user_id": st.session_state.user_id,
            "session_id": st.session_state.session_id,
            "memory_messages": st.session_state.messages
        }

        try:
            # 发起 POST 请求，开启 stream=True
            with requests.post(api_url, json=payload, stream=True) as response:
                if response.status_code != 200:
                    st.error(f"❌ 接口请求失败: HTTP {response.status_code}")
                else:
                    # 解析 SSE 数据流
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')

                            # 过滤掉 SSE 协议中的 data: 前缀
                            if decoded_line.startswith("data: "):
                                json_str = decoded_line[6:]  # 截取 'data: ' 之后的内容

                                try:
                                    data = json.loads(json_str)
                                    chunk_type = data.get("chunk_type")

                                    # 状态提示处理
                                    if chunk_type == "ai_status":
                                        status_placeholder.caption(f"⏳ 状态: {data.get('ai_status')}")
                                    elif chunk_type == "rag_docs_count":
                                        status_placeholder.caption(f"📚 检索到文档数量: {data.get('rag_docs_count')}")

                                    # 文本内容处理
                                    elif chunk_type == "messages" or chunk_type == "fast_reply":
                                        content = data.get("content", "")
                                        full_response += content
                                        # 实时更新文本，并添加闪烁的光标效果
                                        message_placeholder.markdown(full_response + " ▌")

                                    # 结束信号处理
                                    elif chunk_type == "stop_messages":
                                        break

                                except json.JSONDecodeError:
                                    continue  # 忽略解析错误的行

            # 流式输出结束后，清理光标并定型文本
            message_placeholder.markdown(full_response)
            status_placeholder.empty()  # 清除顶部的状态提示

            # 将 AI 回复保存到历史记录中 (注意你的后端要求 role 为 "ai")
            st.session_state.messages.append({"role": "ai", "content": full_response})

        except requests.exceptions.ConnectionError:
            st.error("❌ 无法连接到服务器，请确保 FastAPI 后端已启动并在正确的端口运行。")
        except Exception as e:
            st.error(f"❌ 发生未知错误: {str(e)}")