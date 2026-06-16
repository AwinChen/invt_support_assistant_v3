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
