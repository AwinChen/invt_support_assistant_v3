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

class PSTools:



if __name__ == '__main__':
    # 调用上传方法
    url = PSTools.save_fig_to_minio(fig)
    print("图片访问地址：", url)