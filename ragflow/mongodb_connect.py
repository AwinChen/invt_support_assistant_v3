import os

import pandas as pd
from pymongo import MongoClient



class mongodb_operations():
    def __init__(self):
        self.mongodb_config = {
            "db_name": os.getenv("MONGO_DATABASE"),
            "collection_name": "es_inserted_info"
        }

        self.client = MongoClient(os.getenv("MONGO_URL"))
        self.db = self.client[self.mongodb_config['db_name']]
        self.collection = self.db[self.mongodb_config['collection_name']]

    def insert_dataframe_to_mongodb(self, df):
        """将 DataFrame 数据插入 MongoDB 指定集合"""
        try:
            # 转换 DataFrame 为字典列表
            records = df.to_dict('records')
            if not records:
                print("DataFrame 为空，未插入数据")
                return

            # 插入数据
            result = self.collection.insert_many(records)
            print(f"成功插入 {len(result.inserted_ids)} 条记录")
        except Exception as e:
            print(f"插入数据时出错: {e}")
        finally:
            # 关闭连接
            if self.client:
                self.client.close()

    def delete_all_data_in_mongodb(self):
        """将数据全部删除"""
        try:
            result = self.collection.delete_many({})
            print(f"成功删除 {result.deleted_count} 条记录")
        except Exception as e:
            print(f"删除数据时出错: {e}")
        finally:
            # 关闭连接
            if self.client:
                self.client.close()

    def delete_specify_data_in_mongodb(self, query):
        """指定数据清除，需要自行指定query，query格式按照mongodb标准写入"""
        try:
            result = self.collection.delete_many(query)
            print(f"成功删除 {result.deleted_count} 条记录")
        except Exception as e:
            print(f"删除数据时出错: {e}")
        finally:
            # 关闭连接
            if self.client:
                self.client.close()

    def delete_selected_data_by_doc_name_in_mongodb(self, doc_name):
        """指定doc_name，清除其有关的全部数据"""
        try:
            searched_result = self.collection.find({"doc_name": {"$regex": doc_name}})
            selected_list_for_delete = []
            for idx in searched_result:
                selected_list_for_delete.append(idx['doc_name'])
            selected_list_for_delete = list(set(selected_list_for_delete))
            # print(selected_list_for_delete)

            for doc in selected_list_for_delete:
                print(f"正在删除 {doc} ...")
                result = self.collection.delete_many({"doc_name": doc})
                print(f"成功删除 {result.deleted_count} 条记录")
        except Exception as e:
            print(f"删除数据时出错: {e}")
        finally:
            # 关闭连接
            if self.client:
                self.client.close()

    def find_es_id_by_doc_name(self, doc_name):
        try:
            searched_result = self.collection.find({"doc_name": {"$regex": doc_name}})
            selected_es_id_list = []
            for idx in searched_result:
                selected_es_id_list.append(idx['es_id'])
            return selected_es_id_list
        except Exception as e:
            print(f"删除数据时出错: {e}")
        finally:
            # 关闭连接
            if self.client:
                self.client.close()


if __name__ == "__main__":
    # mongodb_operations().delete_selected_data_by_doc_name_in_mongodb("主文档")
    # mongodb_operations().delete_specify_data_in_mongodb(query={"es_id": "74dd1636-5316-11f0-83e4-8c32230698f2"})
    mongodb_operations().delete_all_data_in_mongodb()
