import json
import operator
import sys

import pandas as pd
from dotenv import load_dotenv
import os
import asyncio
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.rag_agent_async import create_rag_agent
from utils.tools import Tools

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
from typing import Annotated, List, Tuple, Union, Literal, Dict, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode, InjectedState
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command, interrupt
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, RemoveMessage

from prompt.prompt_for_prod_select_agent import *

import aiomysql

# import pymysql
# from pymysql.connections import Connection


load_dotenv()
VBASE_URL = os.getenv("VBASE_URL")
VMODEL_NAME = os.getenv("VMODEL_NAME")
VAPI_KEY = os.getenv("VAPI_KEY")

MYSQL_URL = os.getenv("MYSQL_URL")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_CONFIG = {
    "host": MYSQL_URL.split(":")[0],
    "port": int(MYSQL_URL.split(":")[1]),
    "user": MYSQL_USER,
    "password": MYSQL_PASSWORD,
    "db": MYSQL_DATABASE,
    "charset": "utf8mb4",
    "autocommit": True,
}


# 获取异步流式输出写入器
async def get_async_stream_writer():
    return get_stream_writer()


product_category_mapping = {
    "PLC": {
        "小型PLC": ["IVC1L系列", "TS600系列"],
        "中型PLC": ["TM700系列", "AX系列"],
        "大型PLC": ["TP6000系列"]
    }
}


def build_product_tree_text(mapping: dict) -> str:
    lines = []
    for lv1, lv2_dict in mapping.items():
        lines.append(f"一级大类：{lv1}")
        for lv2, series_list in lv2_dict.items():
            lines.append(f"  二级类别：{lv2}")
            lines.append(f"    产品系列：{'、'.join(series_list)}")
    return "\n".join(lines)


class Agentstate(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    cache: Annotated[Dict, operator.or_]  # 代理中间缓存


# 产品参数匹配可行域
class IntCompareSpec(BaseModel):
    value: Optional[int] = Field(
        None,
        description="比较的数值本体，例如 2、4、8"
    )
    compare: Optional[Literal[">=", "<="]] = Field(
        default=">=",
        description=(
            "比较符。"
            "工业产品选型默认采用宽松匹配："
            "用户描述通常理解为“至少满足”。"
            "仅允许 >= 或 <=。"
        )
    )


class PLCSpecs(BaseModel):
    reasoning: str = Field(
        description=(
            "用于生成你向用户推荐选型参数的分析理由，需输出具体逻辑判断，不输出模板、不输出占位符、不输出内部推理过程。"
        )
    )

    # series: Optional[list[Literal["IVC1L系列", "TS600系列", "TM700系列", "AX系列", "TP6000系列"]]] = Field(
    #     default=None,
    #     description=(
    #     "用户匹配到的产品系列列表。"
    #     "只能依据用户原文中明确出现的系列名称填写，不允许基于行业常识、上下文或语义推断。"
    #     "当用户一次提及多个系列时，需全部返回。"
    #     "若用户未明确提及任何系列，则返回 null。"
    # )
    # )

    EtherCAT_bus_axis_number: Optional[IntCompareSpec] = Field(
        None,
        description="EtherCAT总线控制轴数，支持比较符判断；轴数类字段默认倾向 >=。"
    )

    RS232_number: Optional[IntCompareSpec] = Field(
        None,
        description="串口RS232的数量，支持比较符判断；数量类字段默认倾向 >=。"
    )

    ethernet_protocol: Optional[List[Literal["EtherNet/IP", "Modbus TCP", "TCP/IP", "UDP", "OPC UA", "CANopen"]]] \
        = Field(None, description="以太网协议，可多选：'EtherNet/IP'、'Modbus TCP'、'TCP/IP'、'UDP'、'OPC UA'、'CANopen'")

    expansion_card: Optional[List[Literal["CAN", "4G", "WIFI"]]] \
        = Field(None, description="扩展卡，可多选：CAN、4G、WIFI")

    programming_language: Optional[
        List[Literal["梯形图(LD)", "顺序功能图(SFC)", "指令表(IL)", "C语言", "结构化文本(ST)", "连续功能图"]]] \
        = Field(None,
                description="编程语言，可多选：梯形图(LD)、顺序功能图(SFC)、指令表(IL)、C语言、结构化文本(ST)、连续功能图")

    motion_control_function: Optional[List[Literal["PTO", "PTP", "电子凸轮", "CNC", "ROBOT"]]] \
        = Field(None, description="运动控制功能，可多选：PTO、PTP、电子凸轮、CNC、ROBOT")

    # RS485_number: Optional[int] = Field(None, description="串口RS485的数量，支持比较符判断")
    # rated_voltage: Optional[Literal["24V", "220V"]] = Field(None, description="额定电压，仅允许：24V、220V")
    # local_pulse_axis_count: Optional[IntCompareSpec] = Field(None, description="本地脉冲轴数量，支持比较符判断")
    # ethercat_master_count: Optional[IntCompareSpec] = Field(None, description="EtherCAT主站数量，支持比较符判断")
    # ethercat_ring_control: Optional[Literal["Y", "N"]] = Field(None, description="EtherCAT环网控制，如：Y, N")
    # ethernet_count: Optional[IntCompareSpec] = Field(None, description="EtherNET数量，支持比较符判断")
    # high_speed_counter: Optional[IntCompareSpec] = Field(None, description="高速计数器数量，支持比较符判断")
    # analog_io: Optional[str] = Field(None, description="本体模拟量数量，如：2AI;1AO")
    # temperature_detect: Optional[str] = Field(None, description="本体温度检测数量，如：2路热电阻")
    # high_speed_input: Optional[IntCompareSpec] = Field(None, description="本体高速输入数量，支持比较符判断")
    # high_speed_output: Optional[IntCompareSpec] = Field(None, description="本体高速输出数量，支持比较符判断")
    # io_output_type: Optional[str] = Field(None, description="本体I/O输出类型，如：继电器输出、漏型NPN、源型PNP")

    # ethernet_protocol: Optional[str] = Field(None, description="以太网协议")
    # operating_temperature: Optional[str] = Field(None, description="运行环境温度(℃)")
    # protection_level: Optional[str] = Field(None, description="防护等级，如：IP20")
    # installation_method: Optional[str] = Field(None, description="安装方式")
    # surge_kv: Optional[str] = Field(None, description="浪涌(kV)")
    # static_level: Optional[str] = Field(None, description="静电等级")
    # safety_cert: Optional[str] = Field(None, description="安全认证，如：CE")


# =========================
# 构图节点
# =========================
async def init(state: Agentstate):
    return {
        "cache": {
            "fake_stream_text": '',
            "current_node":'',
            "selected_category_lv1": '',
            "selected_category_lv2": '',
            "selected_specs": {},  # 存放当前抽取到的参数状态字典
            "selected_specs_to_text": '',
            "need_confirm": False,  # 标记用户是否已经确认
            "selected_product": '',  # 选取的产品型号详情
        }
    }

async def _compress_messages(input_dict: dict, config: RunnableConfig):
    """在 chain 内对 messages 做压缩，返回压缩后的输入 dict"""
    messages = input_dict["messages"]
    llm_nothink = config['configurable']['llm_nothink']

    dialogue_text = Tools.format_messages(messages)
    if len(dialogue_text) >= 3500:
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
        summary = await llm_nothink.ainvoke(summary_msg)
        human_msg = HumanMessage(content=messages[-1].content)
        compressed = [
            SystemMessage(content=f"历史对话摘要：\n{summary.content}"),
            human_msg
        ]
        return {"messages": compressed}
    return input_dict   # 没超限，原样透传


async def rag_pipeline(state: Agentstate, config: RunnableConfig):
    rag_agent = await create_rag_agent()
    agent_state = {
        "messages": state["messages"],
    }
    rag_agent_state = await rag_agent.ainvoke(agent_state, {"recursion_limit": 15})

    markdown = rag_agent_state['cache']['markdown']
    rag_prompt = RAG_PROMPT.replace("{rag_context}", markdown)

    # 构建仅供rag调用的临时消息
    sys_msg = SystemMessage(content=rag_prompt)
    temp_messages = [sys_msg] + state["messages"]
    llm = config['configurable']['llm']

    response = await llm.ainvoke(temp_messages)

    # 清除response中think内容以压缩上下文
    if "<think>" or "</think>" in response.content:  # 清除response中think内容以压缩上下文
        content = re.sub("<think>.*?</think>", '', response.content, flags=re.DOTALL)
        response = AIMessage(content=content)

    messages = state["messages"]
    messages.append(response)

    return {
        "messages": messages
    }



async def after_rag_pipeline(state: Agentstate, config: RunnableConfig):
    messages = state["messages"]

    response_text = "\n\n当前正在产品选型，**回复【1】继续选型** 或 **回复【0】退出选型**，如有疑问请随时补充。"
    user_reply = interrupt({
        "status": "waiting_for_confirmation",
        "response_text": response_text,
    })

    messages.extend([AIMessage(content=response_text), HumanMessage(content=user_reply)])

    if user_reply.strip() in ['1', '【1】']: # 回复【1】继续选型
        return Command(
            update={'messages': messages},
            goto=state["cache"]["current_node"]
        )

    elif user_reply.strip() in ['0', '【0】', '结束选型', '退出', '取消']:
        fake_stream_text = "已结束产品选型，如您后续仍有选型需求，可直接回复【我要选型】发起产品选型流程。"
        return Command(
            update={
                "cache": {
                    "fake_stream_text": fake_stream_text,
                }
            },
            goto=END
        )

    else:
        return Command(
            update={'messages': messages},
            goto="rag_pipeline"
        )

async def confirm_category(state: Agentstate, config: RunnableConfig):
    messages = state["messages"]
    selected_category_lv1 = 'PLC'  # 由于目前只开放PLC选型功能，这里写死
    # selected_category_lv1 = state['cache']['selected_category_lv1']
    selected_category_lv2 = state['cache']['selected_category_lv2']

    response_text = CONFIRM_CATEGORY_REPLY
    user_reply = interrupt({
        "status": "waiting_for_confirmation",
        "response_text": response_text,
    })
    messages.extend([AIMessage(content=response_text), HumanMessage(content=user_reply)])


    if user_reply.strip() in ['1', '【1】', '小型PLC']:
        selected_category_lv2 = '小型PLC'
    elif user_reply.strip() in ['2', '【2】', '中型PLC']:
        selected_category_lv2 = '中型PLC'
    elif user_reply.strip() in ['3', '【3】', '大型PLC']:
        selected_category_lv2 = '大型PLC'
    elif user_reply.strip() in ['4', '【4】']:
        response_text = "请描述您的应用场景或需求，由小英为您评估推荐"
        user_reply = interrupt({
            "status": "waiting_for_confirmation",
            "response_text": response_text,
        })
        messages.extend([AIMessage(content=response_text), HumanMessage(content=user_reply)])

        return Command(
            update={
                "cache": {
                    "current_node": "confirm_category"
                },
                'messages': messages
            },
            goto='rag_pipeline'
        )

    elif user_reply.strip() in ['0', '【0】', '结束选型', '退出', '取消']:
        fake_stream_text = "已结束产品选型，如您后续仍有选型需求，可直接回复【我要选型】发起产品选型流程。"
        return Command(
            update={
                "cache": {
                    "fake_stream_text": fake_stream_text,
                }
            },
            goto=END
        )

    else:
        return Command(
            goto="confirm_category"
        )

    return Command(
        update={
            "cache": {
                "selected_category_lv1": selected_category_lv1,
                "selected_category_lv2": selected_category_lv2,
            }
        },
        goto="confirm_specs"
    )

async def confirm_specs(state: Agentstate, config: RunnableConfig):
    messages = state["messages"]

    response_text = CONFIRM_SPECS_REPLY
    user_reply = interrupt({
        "status": "waiting_for_confirmation",
        "response_text": response_text,
    })

    messages.extend([AIMessage(content=response_text), HumanMessage(content=user_reply)])

    if user_reply.strip() in ['1', '【1】']: # 如您暂时无法选择并进一步咨询，请回复【1】
        response_text = "请描述您的应用场景或需求，由小英为您评估推荐"
        user_reply = interrupt({
            "status": "waiting_for_confirmation",
            "response_text": response_text,
        })
        messages.extend([AIMessage(content=response_text), HumanMessage(content=user_reply)])

        return Command(
            update={
                "cache": {
                    "current_node": "confirm_specs"
                },
                'messages': messages
            },
            goto='rag_pipeline'
        )
    elif user_reply.strip() in ['0', '【0】', '结束选型', '退出', '取消']: # 如需退出产品选型流程，请回复【0】
        fake_stream_text = "已结束产品选型，如您后续仍有选型需求，可直接回复【我要选型】发起产品选型流程。"
        return Command(
            update={
                "cache": {
                    "fake_stream_text": fake_stream_text,
                }
            },
            goto=END
        )
    else:
        return Command(
            update={
                'messages': messages
            },
            goto='extract_specs'
        )

async def extract_specs(state: Agentstate, config: RunnableConfig):
    selected_category_lv1 = state["cache"]["selected_category_lv1"]
    selected_category_lv2 = state["cache"]["selected_category_lv2"]
    product_tree_text = build_product_tree_text(product_category_mapping)

    prompt = ChatPromptTemplate.from_messages([
        ("system", PLC_SPECS_EXTRACT_PROMPT),
        MessagesPlaceholder("messages"),
    ])

    chain = RunnableLambda(_compress_messages) | prompt | config['configurable']['llm'].with_structured_output(PLCSpecs)

    result = await chain.ainvoke({"messages": state["messages"]})
    if not result:  # 严格格式化输出，避免程序中断
        raise ValueError("解析失败：extract_specs回复不符合格式化输出！")
    selected_specs = result.model_dump()
    selected_specs['categories'] = selected_category_lv2

    return {
        "cache": {
            "selected_specs": selected_specs
        }
    }

async def check_specs(state: Agentstate):
    messages = state["messages"]
    selected_specs = state["cache"]["selected_specs"]

    def build_selected_specs_to_text(selected_specs):
        recommand_specs_mapping = {
            'categories': "产品类别",
            # 'series': '产品系列',
            'EtherCAT_bus_axis_number': 'EtherCAT总线控制轴数',
            'RS232_number': '串口RS232的数量',
            'ethernet_protocol': '以太网协议',
            'expansion_card': '扩展卡',
            'programming_language': '编程语言',
            'motion_control_function': '运动控制功能',
            # "reasoning": '**参数推荐理由**'
        }
        lines = []

        for key in recommand_specs_mapping.keys():

            # selected_specs 没有该字段时跳过
            if key not in selected_specs:
                continue

            value = selected_specs[key]

            # 跳过字段
            if key in ["need_confirm"]:
                continue

            # 空值
            if value in [None, "", [], {}]:
                value_text = "(不做要求)"

            # IntCompareSpec
            elif isinstance(value, dict) and "value" in value:
                compare = value.get("compare", "")
                val = value.get("value", "")
                value_text = f"{compare}{val}"

            # list
            elif isinstance(value, list):
                value_text = "、".join(map(str, value))

            else:
                value_text = str(value)

            lines.append(f"- {recommand_specs_mapping[key]}：{value_text}")

        return "\n".join(lines)

    selected_specs_to_text = build_selected_specs_to_text(selected_specs)

    fake_stream_text = (
        "系统已结合您的需求描述，初步推荐以下选型参数。请您确认是否符合需求：\n\n"
    )

    fake_stream_text += selected_specs_to_text

    fake_stream_text += (
        "\n\n接下来，您可以:"
        "\n\n**如需更改参数，可回复【1】；**"
        "\n\n**如需无需修改，回复【2】；**"
        "\n\n**如需进一步咨询，回复【3】；**"
        "\n\n**如需退出选型环节，回复【0】；**"
    )

    # 1. 触发动态中断！
    # 程序运行到这里会挂起，并将字典内的状态返回给前端/API
    # 等待外部传入 resume 数据后，user_reply 就会被赋值为用户的输入
    response_text = fake_stream_text
    user_reply = interrupt({
        "status": "waiting_for_confirmation",
        "response_text": response_text,
    })
    messages.extend([AIMessage(content=response_text), HumanMessage(content=user_reply)])

    if user_reply.strip() in ['1', '【1】']: # 如需更改参数，可回复【1】
        response_text = "请您描述需要修改的内容，小英将为您进行更改"
        user_reply = interrupt({
            "status": "waiting_for_confirmation",
            "response_text": response_text,
        })
        messages.extend([AIMessage(content=response_text), HumanMessage(content=user_reply)])

        return Command(
            update={
                'messages': messages
            },
            goto='extract_specs'
        )

    elif user_reply.strip() in ['2', '【2】']: # 如需无需修改，回复【2】
        return Command(
            update={
                "messages": messages,
                "cache": {
                    "selected_specs": selected_specs,
                    "selected_specs_to_text": selected_specs_to_text,
                }
            },
            goto="search_db"
        )
    elif user_reply.strip() in ['3', '【3】']: # 如需进一步咨询，回复【3】
        response_text = "请描述您的应用场景或需求，由小英为您评估推荐"
        user_reply = interrupt({
            "status": "waiting_for_confirmation",
            "response_text": response_text,
        })
        messages.extend(
            [AIMessage(content=response_text), HumanMessage(content=user_reply)])

        return Command(
            update={
                "cache": {
                    "current_node": "check_specs"
                },
                'messages': messages
            },
            goto='rag_pipeline'
        )

    elif user_reply.strip() in ['0', '【0】']:
        fake_stream_text = "已结束产品选型，如您后续仍有选型需求，可直接回复【我要选型】发起产品选型流程。"
        return Command(
            update={
                "cache": {
                    "fake_stream_text": fake_stream_text,
                }
            },
            goto=END
        )

    else:
        return Command(
            goto="check_specs",
        )

async def search_db(state: Agentstate):
    selected_category_lv1 = state["cache"]["selected_category_lv1"]
    selected_category_lv2 = state["cache"]["selected_category_lv2"]
    selected_specs = state["cache"]["selected_specs"]
    selected_specs_to_text = state["cache"]["selected_specs_to_text"]

    tables_mapping = {
        "PLC": 'PLC_all_series',
    }
    tables = [tables_mapping[selected_category_lv1]]
    all_df = []

    try:
        # 每次动态创建 pool
        mysql_conn = await aiomysql.create_pool(
            minsize=1,
            maxsize=15,
            **MYSQL_CONFIG
        )

        async with mysql_conn.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                for table in tables:
                    query = f"SELECT * FROM `{table}`"
                    await cursor.execute(query)
                    rows = await cursor.fetchall()

                    if rows:
                        df = pd.DataFrame(rows)
                        all_df.append(df)

    except Exception as e:
        raise RuntimeError(f"MySQL查询或数据处理失败：{e}") from e

    df = pd.concat(all_df, ignore_index=True) if all_df else pd.DataFrame()

    # 精筛
    # 定义比较操作符映射
    ops = {
        "=": operator.eq, ">=": operator.ge, "<=": operator.le,
        ">": operator.gt, "<": operator.lt
    }

    def multi_select_overlap_filter(df, column_name, selected_values):

        if not selected_values:
            return df

        if column_name not in df.columns:
            return df

        selected_set = {
            str(p).strip()
            for p in selected_values
        }

        def check_overlap(db_item):

            # None / NaN / 空字符串
            if pd.isna(db_item) or str(db_item).strip() == "":
                return False

            db_set = {
                p.strip()
                for p in str(db_item).split(";")
                if p.strip()
            }

            return any(p in db_set for p in selected_set)

        return df[df[column_name].apply(check_overlap)]

    # --- 1. 产品二级类别 ---
    if selected_specs['categories']:
        df = df[df["categories"] == selected_category_lv2]

    # --- 2. 产品系列 ---
    # if selected_specs['series']:
    #     df = df[df["series"].isin(selected_specs['series'])]

    # --- 3. EtherCAT总线控制轴数 (IntCompareSpec 匹配) ---
    if selected_specs['EtherCAT_bus_axis_number']:
        if selected_specs['EtherCAT_bus_axis_number']["value"] is not None:
            val = selected_specs['EtherCAT_bus_axis_number']["value"]
            op = ops.get(selected_specs['EtherCAT_bus_axis_number']["compare"], operator.eq)
            df = df[df["EtherCAT_bus_axis_number"].apply(lambda x: op(int(x), val) if x is not None else False)]

    # --- 4. 串口RS232数量 (IntCompareSpec 匹配) ---
    if selected_specs['RS232_number']:
        # print(selected_specs['RS232_number'])
        if selected_specs['RS232_number']["value"] is not None:
            val = selected_specs['RS232_number']["value"]
            op = ops.get(selected_specs['RS232_number']["compare"], operator.eq)
            df = df[df["RS232_number"].apply(lambda x: op(int(x), val) if x is not None else False)]

    # --- 5. 以太网协议 (字符串多选重叠匹配) ---
    df = multi_select_overlap_filter(
        df,
        "ethernet_protocol",
        selected_specs['ethernet_protocol']
    )

    # --- 6. 扩展卡 (字符串多选重叠匹配) ---
    df = multi_select_overlap_filter(
        df,
        "expansion_card",
        selected_specs['expansion_card']
    )

    # --- 7. 编程语言 (字符串多选重叠匹配) ---
    df = multi_select_overlap_filter(
        df,
        "programming_language",
        selected_specs['programming_language']
    )

    # --- 8. 运动控制功能 (字符串多选重叠匹配) ---
    df = multi_select_overlap_filter(
        df,
        "motion_control_function",
        selected_specs['motion_control_function']
    )

    # 最终查询展示字段
    PLC_FIELD_MAP = {
        "product_id": "产品型号",
        "categories": "产品类别",
        # "series": "产品系列",
        "name": "产品名称",
        # "type": "产品类型",
        # "status": "产品状态",
        # "rated_voltage": "额定电压",

        # "EtherCAT_number": "EtherCAT主站数量",
        # "EtherCAT_ring_control": "EtherCAT环网控制",
        # "EtherNET_number": "以太网口数量",
        "RS232_number": "RS232串口数量",
        # "RS485_number": "RS485串口数量",

        "EtherCAT_bus_axis_number": "EtherCAT总线轴数",
        "motion_control_function": "运动控制功能",

        "ethernet_protocol": "以太网协议",

        # "program_size": "程序容量",
        # "config_platform": "组态平台",
        "programming_language": "编程语言",
        'expansion_card': '扩展卡',
        # "external_dimensions": "外形尺寸"
    }

    if df.empty:
        prod_json_text = "（未匹配到相关产品型号信息。）"

        final_reply = f"""
    当前已根据确认的选型参数：
    \n\n{selected_specs_to_text}
        \n\n当前根据已确认的选型参数，暂未查询到匹配的产品型号。
    可能由于筛选条件较严格，导致暂无符合条件的产品。
    是否可以适当减少部分筛选要求，以便进一步帮助筛选合适产品？
    """
    else:
        prod_df = df[list(PLC_FIELD_MAP.keys())].where(pd.notnull(df), None).rename(columns=PLC_FIELD_MAP).fillna(
            '')
        # prod_md_text = prod_df.to_markdown(index=False)
        prod_json = {}
        for _, row in prod_df.iterrows():
            row_dict = row.to_dict()
            product_id = row_dict.pop("产品型号", None)
            if product_id:
                prod_json[str(product_id)] = row_dict

        prod_json_text = json.dumps(
            prod_json,
            ensure_ascii=False,
            indent=2
        )

        final_reply = f"""
    **当前已根据确认的选型参数**：
    \n\n{selected_specs_to_text}
    \n\n**筛选出产品信息如下**：
    <div style="white-space: pre-wrap;">
    {prod_json_text}
    </div>
    \n\n**已根据确认的选型参数匹配到如下产品型号：{'、'.join(prod_df['产品型号'])}**。如需进一步了解某个产品的详细功能参数、应用场景或选型建议，您可以继续告诉我，如需重新发起产品选型流程，请回复【我要选型】。
    """

    return Command(
        update={
            "cache": {
                "fake_stream_text": final_reply,
                "selected_product": prod_json_text
            }
        },
        goto=END
    )

async def create_prod_select_agent():
    subgraph = StateGraph(Agentstate)

    # nodes
    subgraph.add_node("init", init)

    subgraph.add_node("confirm_category", confirm_category)
    subgraph.add_node("confirm_specs", confirm_specs)
    subgraph.add_node("extract_specs", extract_specs)
    subgraph.add_node("check_specs", check_specs)
    subgraph.add_node("search_db", search_db)
    subgraph.add_node("rag_pipeline", rag_pipeline)
    subgraph.add_node("after_rag_pipeline", after_rag_pipeline)

    # start
    subgraph.add_edge(START, "init")

    # category
    subgraph.add_edge("init", "confirm_category")

    # specs
    subgraph.add_edge("extract_specs", "check_specs")

    subgraph.add_edge("rag_pipeline", "after_rag_pipeline")

    # search_db
    subgraph.add_edge("search_db", END)

    subgraph_compiled = subgraph.compile()

    # graph_png = subgraph_compiled.get_graph().draw_mermaid_png()
    # with open("prod_select_agent.png", "wb") as f:
    #     f.write(graph_png)

    return subgraph_compiled

if __name__ == '__main__':
    llm = ChatOpenAI(
        base_url="http://192.168.5.197:8000/v1",
        model="qwen3_14b",
        api_key="czy0109248",
        max_tokens=6000,
        temperature=0.6,
        top_p=0.95,
        presence_penalty=1.0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0.0,
        },
    )
    llm_nothink = ChatOpenAI(
        base_url="http://192.168.5.197:8000/v1",
        model="qwen3_14b",
        api_key="czy0109248",
        max_tokens=6000,
        temperature=0.7,
        top_p=0.80,
        presence_penalty=1.0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0.0,
        },
    )

    input_state = {
        "messages": [
            HumanMessage(
                content="我有一个设备，10个伺服走Ethercat,3个变频器走RS485，用以太网连接一个触摸屏，编程语言需要支持ST,推荐一个英威腾中型PLC的型号吧")],
    }

    # 同步测试：
    # response = graph.invoke(input_state, config={"recrusion_limit": 2})
    # print(response)

    # 异步测试：
    async def main():
        prod_select_agent = await create_prod_select_agent()
        response = await prod_select_agent.ainvoke(
            input_state,
            config={"recursion_limit": 15},
        )
        print(response['cache'])

    asyncio.run(main())

    # 想要一款PLC，支持24V的变频控制 听完你的介绍，我可能需要中型PLC  EtherCAT轴数至少16个，需要支持扩展口
    # 想要一款中型PLC，支持24V的变频控制  EtherCAT轴数至少16个，需要支持扩展口
    # 想找支持EtherCAT的PLC  中型PLC EtherCAT 至少16个及以上 没有其它参数需求
    # 要一台支持 EtherNet/IP 和 Modbus TCP 的小型PLC 想了解TS635产品详细功能
    # 你好，我这边有个小型设备项目，想选PLC。 需要带以太网通讯，并且支持Modbus TCP。 你先帮我推荐一下适合的PLC方向。

    # 我有个设备项目，需要做PLC产品选型，能帮我推荐下合适的方向吗
    # 项目主要是小型自动化设备，目前倾向于小型PLC方案
    # 另外我这边后续可能需要远程运维，所以PLC最好支持4G扩展卡
    # TS600系列里，我看到有TS634P和TS635这两个型号，两者在性能和功能上有什么区别？
