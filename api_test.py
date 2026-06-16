import requests
import json

def request_faq_search():
    url = f"http://127.0.0.1:8000/chat/faq_search"
    question = {"question": "你好，如何处理单元故障？"}

    response = requests.post(url, json=question)
    output = response.text
    # print(response.text)
    return output

def request_chat_stream(user_input, user_id='001', session_id='1'):
    url = "http://127.0.0.1:8000/chat/stream"
    # url = "http://192.168.5.197:8501/chat/stream"

    payload = {
        # 根据你的ChatRequest模型构造请求体，例如：
        "user_id": user_id,
        "session_id": session_id,
        "memory_messages": [{"role": "user", "content": f"{user_input}"}],
        # ... 其他可能需要的字段，如model, temperature等
    }
    # 设置stream=True以保持连接开放，逐步接收数据块
    response = requests.post(url, json=payload, stream=True)
    for line in response.iter_lines():
        if line:
            # SSE协议数据行通常以"data: "开头
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data: '):
                # 提取JSON数据部分
                json_data = decoded_line[6:]
                try:
                    data_obj = json.loads(json_data)
                    # 打印或处理数据对象
                    print(f"模拟后端接收到数据: {data_obj}")
                    # 如果你期望流式文本，可能数据在data_obj的某个字段里，如"message"或"delta"
                    # 例如：print(data_obj.get("content", ""), end="", flush=True)
                except json.JSONDecodeError as e:
                    print(f"解析JSON失败: {e}, 原始数据: {json_data}")

if __name__ == '__main__':
    # output = request_faq_search()
    while True:
        user_question = input("\n请输入问题：").strip()
        request_chat_stream(user_question, user_id='002', session_id='1')
    # 中型PLC产品选型
    # EtherCAT总线控制轴数要求8个以上
    # EtherCAT总线控制轴数我不确定是多少个，你可以帮我确定一下吗，我需要带5个伺服
