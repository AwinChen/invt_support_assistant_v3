import operator
import time
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

# 1. 定义状态结构，确保包含所有必要字段
class AgentState(TypedDict):
    user_query: str
    ai_response: str  # 用于存放对用户的回复
    updated_config: dict  # 用于存放更新后的配置信息
    # 如果希望收集日志，可以添加如下字段
    # logs: Annotated[list[str], operator.add]

# 2. 定义节点函数
def some_previous_node(state: AgentState):
    """这是一个假设的前置节点，用于处理一些逻辑或 simply pass"""
    # 这里可以有一些处理逻辑，或者直接返回state传递下去
    return state

def reply_to_user(state: AgentState):
    """节点：生成对用户问题的回复"""
    # 你的逻辑 here，例如调用LLM生成回复
    print("执行 reply_to_user")
    time.sleep(5)
    response_content = "这是根据您的问题生成的回复。"  # 替换为实际LLM调用
    new_config = {"ai_response": response_content}
    return {"ai_response": response_content}


def update_configuration(state: AgentState):
    """节点：调用LLM分析并更新用户配置"""
    # 你的逻辑 here，基于当前state（如user_query）调用LLM来更新配置
    print("执行 update_configuration")
    time.sleep(3)
    new_config = {"last_topic": state["user_query"], "updated_at": "2025-09-10"}  # 替换为实际LLM调用和逻辑
    return {"updated_config": new_config}

def aggregate_results(state: AgentState):
    """（可选）聚合节点：当两个并行节点都完成后，可以在这里进行一些后续处理"""
    print(f"回复内容: {state['ai_response']}")
    print(f"更新后的配置: {state['updated_config']}")
    # 可以在这里记录日志等
    # return {"logs": ["Aggregation completed."]}
    return state

# 3. 构建图
builder = StateGraph(AgentState)

# 添加节点
builder.add_node("some_previous_node", some_previous_node)
builder.add_node("reply_to_user", reply_to_user)
builder.add_node("update_configuration", update_configuration)
builder.add_node("aggregate", aggregate_results) # 可选聚合节点

# 设置边，建立并行关系
builder.add_edge(START, "some_previous_node") # 从开始到某个前置节点
# 关键：从前置节点同时引出两条边，分别指向回复节点和配置更新节点
builder.add_edge("some_previous_node", "reply_to_user")
builder.add_edge("some_previous_node", "update_configuration")
# 将两个并行节点的输出汇聚到聚合节点（或者你也可以让它们直接到END）
builder.add_edge("reply_to_user", "aggregate")
builder.add_edge("update_configuration", "aggregate")
builder.add_edge("aggregate", END) # 从聚合节点到结束

# 编译图
graph = builder.compile()

# 4. 调用图
initial_state = {"user_query": "用户当前的问题是什么？"}
result = graph.invoke(initial_state)
print(result)