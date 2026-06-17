import asyncio
import aiomysql
import traceback


async def tes_mysql():
    try:
        conn = await aiomysql.connect(
            host="192.168.5.197",
            port=5455,
            user="root",
            password="infini_rag_flow",
            db="invt_ai",
            charset="utf8mb4",
            autocommit=True,
        )

        print("连接成功")

        async with conn.cursor() as cur:
            await cur.execute("SELECT VERSION()")
            result = await cur.fetchone()
            print(result)

        conn.close()

    except Exception as e:
        print("详细异常：")
        traceback.print_exc()



if __name__ == "__main__":
    asyncio.run(tes_mysql())
(
    ('prod_select_agent:9bd40f5f-fd20-4c2b-320e-c1090424e87e', 'rag_pipeline:61456d82-f8f1-4cf5-e6ee-528b68f316af'),
    'messages',
    (AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': None, 'function': {'arguments': ' 特', 'name': None}, 'type': None}]}, response_metadata={}, id='lc_run--019ed46e-cee3-7440-ba1e-abf65371a4e9', tool_calls=[], invalid_tool_calls=[{'name': None, 'args': ' 特', 'id': None, 'error': None, 'type': 'invalid_tool_call'}], tool_call_chunks=[{'name': None, 'args': ' 特', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]),
     {'thread_id': 'test_user_001_687a3e18-d525-4cc3-a9b1-3dab48c931dd', 'ls_integration': 'langchain_chat_model', 'langgraph_step': 2, 'langgraph_node': 'core_agent', 'langgraph_triggers': ('branch:to:core_agent',), 'langgraph_path': ('__pregel_pull', 'core_agent'), 'langgraph_checkpoint_ns': 'prod_select_agent:9bd40f5f-fd20-4c2b-320e-c1090424e87e|rag_pipeline:61456d82-f8f1-4cf5-e6ee-528b68f316af|core_agent:ca30339c-667f-fba3-fa4d-a4ada8a094da', 'checkpoint_ns': 'prod_select_agent:9bd40f5f-fd20-4c2b-320e-c1090424e87e', 'ls_provider': 'openai', 'ls_model_name': 'qwen3_14b', 'ls_model_type': 'chat', 'ls_temperature': 0.7, 'ls_max_tokens': 2048}))
