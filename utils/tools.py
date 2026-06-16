import os
import uuid
from typing import List
import re
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
import io
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from minio import Minio
from matplotlib.figure import Figure

load_dotenv()

class Tools:

    @staticmethod
    def format_messages(messages):
        context = ''
        for msg in messages:
            context += f"{msg.type}: {msg.content}\n"
        return context

    # data_agent_prompt中间代理聊天记录处理
    @staticmethod
    def process_data_agent_messages(messages:List[BaseMessage]) -> str:
        context = ''
        for msg in messages:
            if isinstance(msg, HumanMessage):
                context += f"Human: {msg.content}\n"
            elif isinstance(msg, SystemMessage):
                context += f"system: {msg.content}\n"
            elif isinstance(msg, ToolMessage):
                context += f"system: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                ai_txt = re.sub("<think>(.?)</think>", "", msg.content)
                context += f"AI: {ai_txt}\n"
        return context

    @staticmethod
    def process_chat_messages(cache_messages: List[BaseMessage]) -> str:
        context = ''
        for msg in cache_messages:
            if isinstance(msg, HumanMessage):
                context += f"Human: {msg.content}\n"
            elif isinstance(msg, SystemMessage):
                context += f"system: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                context += f"AI: {msg.content}\n"
        return context

    @staticmethod
    def save_fig_to_minio(fig: Figure) -> str:
        """
        将 matplotlib 的图像对象 fig 上传到 MinIO，并返回公开访问的 URL。

        :param fig: matplotlib.figure.Figure 图像对象
        :return: 图片的公网或内网访问 URL
        """
        # MinIO 配置（可根据需要改为环境变量或配置文件）
        MINIO_URL = os.getenv("MINIO_URL")
        MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
        MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
        MINIO_BUCKET_NAME = os.getenv("BUCKET_FOR_TEMP_IMAGE")
        USE_HTTPS = False  # 设置为 True 如果你使用的是 https


        # 初始化 MinIO 客户端
        client = Minio(
            endpoint=MINIO_URL,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=USE_HTTPS
        )

        # 保存图像到内存
        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format='png', bbox_inches='tight')
        img_bytes.seek(0)

        # 生成唯一文件名
        filename = f"/images/{uuid.uuid4().hex}.png"

        try:
            # 创建桶（如果不存在）
            if not client.bucket_exists(MINIO_BUCKET_NAME):
                client.make_bucket(MINIO_BUCKET_NAME)

            # 上传图像
            client.put_object(
                bucket_name=MINIO_BUCKET_NAME,
                object_name=filename,
                data=img_bytes,
                length=img_bytes.getbuffer().nbytes,
                content_type="image/png"
            )

            # 返回图片 URL
            print(f"Minio: 已成功存入图片：http://{MINIO_URL}/{MINIO_BUCKET_NAME}/{filename}")
            return f"http://{MINIO_URL}/{MINIO_BUCKET_NAME}/{filename}"
        except Exception as e:
            raise RuntimeError(f"上传图片到 MinIO 失败：{e}")

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    ax.set_title("测试图")

    # 调用上传方法
    url = Tools.save_fig_to_minio(fig)
    print("图片访问地址：", url)