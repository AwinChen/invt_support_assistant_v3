"""
测试checkpoint
"""

# from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import StateGraph, MessagesState, START
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi


# model = ChatOpenAI(
#         base_url="http://192.168.5.197:8000/v1",
#         model="Qwen3_8B",
#         api_key="czy0109248",
#         max_tokens=6000,
#         extra_body={
#             "repetition_penalty": 1.05,
#             "chat_template_kwargs": {"enable_thinking": True}
#         },
#     )

model = llm = ChatTongyi(model="qwen3-8b", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", temperature=0,
                     api_key="sk-db286ac12d9940ddb26dd8413279041a", streaming=True, model_kwargs={"enable_thinking": False})


DB_URI = "mongodb://invt_admin:invt123@192.168.5.197:27017/local?authSource=admin&authMechanism=SCRAM-SHA-256"
with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    graph = builder.compile(checkpointer=checkpointer)


    config = {
        "configurable": {
            "thread_id": "2"
        }
    }
    res = graph.invoke({"messages": [{"role": "user", "content": "我叫什么"}]}, config)
    print(res)

    # a = list(checkpointer.list(config))
    # print(a)

