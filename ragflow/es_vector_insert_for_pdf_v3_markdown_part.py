import json
import os
import re
from io import BytesIO
import numpy as np
import pandas as pd
import torch
import uuid
from rag.nlp import search, rag_tokenizer
from rag.utils import truncate
from FlagEmbedding import FlagModel

from rag.nlp import naive_merge, rag_tokenizer, tokenize_chunks, tokenize_table

from datetime import datetime
import xxhash
from PIL import Image

from ES_connect import ESConnection
ES_cnt = ESConnection()

from raw_file_seg_for_markdown import Markdown

from mongodb_connect import mongodb_operations

from MinIO_connect import Minio_Connection

# 设置显示所有行和列
pd.set_option('display.max_rows', None)    # 显示所有行
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)      # 自动调整列宽，避免换行
pd.set_option('display.max_colwidth', None) # 显示完整列内容



class pdf_chunks_to_es_vector_v3_for_markdown():
    def __init__(self):
        # 切分chunk的大小
        # self.embedder_chunk_size = 512 # 默认
        self.embedder_chunk_size = 512
        # embedding model地址
        # self.embedding_model_filepath = "/home/script/invt_agent_demo/invt_agent_for_linux/embedding_model/bge-m3"
        self.embedding_model_filepath = r"E:\for_git\bge-m3"
        # minio bucket名称
        self.minio_bucket_name = 'invt-agent-images'
        # self.minio_main_folder = 'images'
        self.minio_main_folder = 'file-images' # 正式的图片库

    def chunk(self, filename, binary=None, 
          lang="Chinese", **kwargs):

        is_english = lang.lower() == "english"  # is_english(cks)
        parser_config = kwargs.get(
            "parser_config", {
                "chunk_token_num": 128, "delimiter": "\n!?。；！？", "layout_recognize": "DeepDOC"})
        doc = {
            "docnm_kwd": filename,
            "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
        }
        doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
        res = []
        pdf_parser = None

        markdown_parser = Markdown(int(parser_config.get("chunk_token_num", 128)))
        sections, tables = markdown_parser(filename, binary)
        # print(sections)
        # print(len(sections))

        # 测试
        # sections = [sections[8]] # error_key: 8, 10
        # print(sections)

        # 检测原路径使用的主要分隔符
        if "\\" in filename and "/" not in filename:
            separator = "\\"  # 原路径使用反斜杠
        elif "/" in filename and "\\" not in filename:
            separator = "/"   # 原路径使用正斜杠
        else:
            # 混合使用或没有分隔符，默认使用系统分隔符
            separator = os.sep
        pics_filepath = separator.join(re.split(r"[\\/]", filename)[:-1])

        res = tokenize_table(tables, doc, is_english)
       
        chunks = naive_merge(
            sections, int(parser_config.get(
                "chunk_token_num", 128)), parser_config.get(
                "delimiter", "\n!?。；！？"))
        # print(chunks)
        # print(len(chunks))

        # 清除掉为空的chunk
        chunks = [item for item in chunks if item != '']
        # print(chunks)
        # for ck in chunks:
        #     print(ck)
        #     print('\n')

        if kwargs.get("section_only", False):
            return chunks

        res.extend(tokenize_chunks(chunks, doc, is_english, pdf_parser))
        # print(res)
        # print(len(res))
        # for p in res:
        #     print(p)
        #     print('\n')

        # # 修饰全部的chunk
        # final_res = []            
        # for pt in res:
        #     # 如果存在图片格式，则访问本地图片并放入，新增属性image，若没有则不添加image属性
        #     pt1 = self.extract_and_add_multiple_image_info(pt, pics_filepath)
        #     # print(pt1)
        #     # 检测每个chunk，将带有图片格式的从原来的key中移除
        #     pt2 = self.remove_image_references(pt1)
        #     # print(pt2)
        #     # 处理完成后放入最终结果
        #     final_res.append(pt2)
        # # print(final_res)
        # # print(len(final_res))
        # # for p in final_res:
        # #     print(p)
        # #     print('\n')

        return res
        # return final_res

    # 检测并添加image属性
    def extract_and_add_multiple_image_info(self, data_dict, filepath):
        """
        从字典的content_with_weight字段中提取所有图片信息，并添加到字典的image键中
        
        参数:
            data_dict: 包含content_with_weight等字段的字典
            
        返回:
            更新后的字典
        """
        result_dict = data_dict.copy()
        content_with_weight = result_dict.get('content_with_weight', '')
        
        # 使用正则表达式匹配所有 [](images/...jpg) 格式的图片引用
        pattern = r'\[\]\(images/(.*?\.jpg)\)'
        img_name_lst = re.findall(pattern, content_with_weight)
        
        # 将所有匹配的图片信息添加到字典中
        if img_name_lst:
            # 在本地资料夹中寻找相应的图片
            img_filepath = os.path.join(filepath, 'images', img_name_lst[0])
            try:
                if os.path.isfile(img_filepath):
                    img = Image.open(img_filepath).convert('RGB')
            except Exception as e:
                pass
            result_dict['image'] = img
        
        return result_dict
    
    # 检测每个chunk，将带有图片格式的从原来的key中移除
    def remove_image_references(self, data_dict):
        """
        从字典的 content_with_weight, content_ltks, content_sm_ltks 字段中删除图片引用标记。

        参数:
            data_dict: 包含需要清理字段的字典。

        返回:
            修改后的字典。
        """
        # 创建字典的副本以避免修改原始数据
        result_dict = data_dict.copy()
        
        # 1. 处理 content_with_weight - 删除 [](images/...jpg) 格式
        content_with_weight = result_dict.get('content_with_weight', '')
        # 匹配 [](images/...jpg) 模式并将其替换为空字符串
        pattern_with_weight = r'\[\]\(images/.*?\.jpg\)'
        cleaned_content_with_weight = re.sub(pattern_with_weight, '', content_with_weight)
        result_dict['content_with_weight'] = cleaned_content_with_weight
        
        # 2. 处理 content_ltks - 删除图片相关的令牌序列 (imag ... jpg)
        content_ltks = result_dict.get('content_ltks', '')
        # 匹配以 'imag' 开头，后跟一些令牌，以 'jpg' 结尾的模式
        # 使用捕获组来匹配整个序列
        pattern_ltks = r'imag\s+\S+\s+jpg'  # 匹配 "imag", 然后是非空白字符（哈希），然后是 "jpg"
        cleaned_content_ltks = re.sub(pattern_ltks, '', content_ltks)
        # 也可能需要处理可能多余的空格
        cleaned_content_ltks = ' '.join(cleaned_content_ltks.split())  # 移除多余空格
        result_dict['content_ltks'] = cleaned_content_ltks
        
        # 3. 处理 content_sm_ltks - 同样删除图片相关的令牌序列 (imag ... jpg)
        content_sm_ltks = result_dict.get('content_sm_ltks', '')
        pattern_sm_ltks = r'imag\s+\S+\s+jpg'  # 与 content_ltks 相同的模式
        cleaned_content_sm_ltks = re.sub(pattern_sm_ltks, '', content_sm_ltks)
        cleaned_content_sm_ltks = ' '.join(cleaned_content_sm_ltks.split())  # 移除多余空格
        result_dict['content_sm_ltks'] = cleaned_content_sm_ltks
        
        return result_dict

    def index_name(self, uid): return f"invt_{uid}"

    # 初始化向量库
    def init_kb(self, row, vector_size: int):
        idxnm = self.index_name(row["tenant_id"])
        return ES_cnt.createIdx(idxnm, row.get("kb_id", ""), vector_size)

    def embedding(self, docs, mdl, parser_config=None):
        if parser_config is None:
            parser_config = {}
        batch_size = 16

        # 文本预处理
        tts, cnts = [], []
        for d in docs:
            tts.append(d.get("docnm_kwd", "Title"))
            c = "\n".join(d.get("question_kwd", []))
            if not c:
                c = d["content_with_weight"]
            c = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", c)
            if not c:
                c = "None"
            cnts.append(c)

        # 标题嵌入生成
        if len(tts) == len(cnts):
            vts = mdl.encode(tts[0: 1])
            tts = np.concatenate([vts for _ in range(len(tts))], axis=0)

        # 内容嵌入分批生成
        cnts_ = np.array([])
        for i in range(0, len(cnts), batch_size):
            vts = mdl.encode([truncate(c, self.embedder_chunk_size-10) for c in cnts[i: i + batch_size]])
            if len(cnts_) == 0:
                cnts_ = vts
            else:
                cnts_ = np.concatenate((cnts_, vts), axis=0)
        cnts = cnts_

        # 标题与内容向量加权融合​
        title_w = float(parser_config.get("filename_embd_weight", 0.1))
        vects = (title_w * tts + (1 - title_w) *
                cnts) if len(tts) == len(cnts) else cnts

        # 向量存储
        assert len(vects) == len(docs)
        for i, d in enumerate(docs):
            v = vects[i].tolist()
            d["q_%d_vec" % len(v)] = v

    # 加入minio模块，对chunk进一步扩展
    def chunks_increase(self, raw_chunks):
        """
        raw_chunks: 初始切分好的chunks
        """

        docs = []
        # doc = {
        #     "doc_id": task["doc_id"],
        #     "kb_id": str(task["kb_id"])
        # }

        # 处理每个带有图片的chunk，将图片加入minio中
        for chunk in raw_chunks:
            # print(chunk)
            chunk["id"] = xxhash.xxh64((chunk["content_with_weight"]).encode("utf-8")).hexdigest()
            chunk["create_time"] = str(datetime.now()).replace("T", " ")[:19]
            chunk["create_timestamp_flt"] = datetime.now().timestamp()
            if not chunk.get("image"):
                _ = chunk.pop("image", None)
                chunk["img_id"] = ""
                docs.append(chunk)
                # print(chunk)
                # print('\n')
                continue

            output_buffer = BytesIO()
            if isinstance(chunk["image"], bytes):
                output_buffer = BytesIO(chunk["image"])
            else:
                chunk["image"].save(output_buffer, format='JPEG')
            
            # 放进指定minio的bucket中
            img_minio_path = self.minio_main_folder + '/' + chunk["id"] + '.png'
            # # 上抛至minio
            # Minio_Connection().put(self.minio_bucket_name, img_minio_path, output_buffer.getvalue())
            
            chunk["img_id"] = self.minio_bucket_name + '/' + img_minio_path
            del chunk["image"]
            # print(chunk)
            # print('\n')
            docs.append(chunk)

        return docs

    def do_handle_task(self, task):
        task_tenant_id = task["tenant_id"]
        task_document_name = task["name"]
        # print(task_document_name)

        # 初始化
        # try:
        #     # bind embedding model
        #     embedding_model = FlagModel(self.embedding_model_filepath,
        #                                 query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
        #                                 use_fp16=torch.cuda.is_available())
        #
        #     vts = embedding_model.encode(["ok"])
        #     vector_size = len(vts[0])
        # except Exception as e:
        #     error_message = f'Fail to bind embedding model: {str(e)}'
        #     raise Exception(error_message)
        # self.init_kb(task, vector_size)
        
        # Standard chunking methods
        # 根据放入文件类型，选择合适的chunks生成py档案执行
        chunks = self.chunk(task_document_name)
        # print(chunks)
        # print(len(chunks))
        # for ck in chunks:
        #     print(ck)
        #     print('\n')

        # chunks配置上minio相关信息
        chunks = self.chunks_increase(chunks)
        # print(chunks)
        # print(len(chunks))
        # for ck in chunks:
        #     print(ck)
        #     print('\n')

        # # 测试metadata使用
        # from generate_metadata_to_prompts import chunks_summary_to_metadata_by_markdown
        # chunks_summary_to_metadata_by_markdown().metadata_to_prompts(chunks)

        if chunks is None:
            return
        if not chunks:
            return

        # embedding模型生成向量
        # try:
        #     self.embedding(chunks, embedding_model)
        # except Exception as e:
        #     error_message = "Generate embedding error:{}".format(str(e))
        #     raise Exception(error_message)

        df_info_for_db_and_es = pd.DataFrame()
        es_id_lst = []

        doc_store_result = ""
        es_bulk_size = 4
        
        chunks_txt_list = []

        for b in range(0, len(chunks), es_bulk_size):
            # 在名为chunks的dict插入"id"：uid，uid基于uuid.uuid1()，其随着逐渐增加，范围为从0到chunks的长度
            # 需要在chunks的key中无id的时才需添加，如果存在跳过此步骤
            batch = chunks[b:b + es_bulk_size]  # 获取当前批次
            for i, ck in enumerate(batch):
                if "id" in ck.keys():
                    pass
                else:
                    # 生成基于时间戳的UUID (v1版本)
                    uid = uuid.uuid1()
                    # 在字典中插入ID字段
                    ck["id"] = str(uid)
                chunks_txt_list.append(ck)
                es_id_lst.append(ck["id"])

            # 插入ES向量库
            # doc_store_result = ES_cnt.insert(chunks[b:b + es_bulk_size], search.index_name(task_tenant_id))
            # if doc_store_result:
            #     error_message = f"Insert chunk error: {doc_store_result}, please check log file and Elasticsearch!"
            #     raise Exception(error_message)
        
        # # 备用，输出所有chunk的txt用于检查
        # chunks_log_filepath = os.path.join(os.path.dirname(fnm), 'chunks_log.txt')
        # # print(chunks_log_filepath)
        # with open(chunks_log_filepath,  "w", encoding="utf-8") as file:
        #     for cks in chunks_txt_list:
        #         file.write(str(cks) + "\n\n")

        df_info_for_db_and_es['doc_name'] = [task_document_name] * len(es_id_lst)
        df_info_for_db_and_es['es_tabel_index'] = [search.index_name(task_tenant_id)] * len(es_id_lst)
        df_info_for_db_and_es['es_storage_id'] = es_id_lst
        return df_info_for_db_and_es

    def markdown_to_chunks(self, task):
        task_tenant_id = task["tenant_id"]
        task_document_name = task["name"]
        # print(task_document_name)

        # 初始化
        # try:
        #     # bind embedding model
        #     embedding_model = FlagModel(self.embedding_model_filepath,
        #                                 query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
        #                                 use_fp16=torch.cuda.is_available())
        #
        #     vts = embedding_model.encode(["ok"])
        #     vector_size = len(vts[0])
        # except Exception as e:
        #     error_message = f'Fail to bind embedding model: {str(e)}'
        #     raise Exception(error_message)
        # self.init_kb(task, vector_size)

        # Standard chunking methods
        # 根据放入文件类型，选择合适的chunks生成py档案执行
        chunks = self.chunk(task_document_name)
        # print(chunks)
        # print(len(chunks))
        # for ck in chunks:
        #     print(ck)
        #     print('\n')

        # chunks配置上minio相关信息
        chunks = self.chunks_increase(chunks)

        return chunks



if __name__ == "__main__":

    fnm = r'D:\python object\知识图谱脚本\files\TP2000系列可编程控制器用户手册_V1.0.md'

    do_task_msgs = {
        "tenant_id": "plc_product_series",
        "name": fnm,
    }

    chunks = pdf_chunks_to_es_vector_v3_for_markdown().markdown_to_chunks(do_task_msgs)
    print(chunks)

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )


    # print(df_for_storage)

    # 将插入es向量库的信息放入mongodb数据库
    # mongodb_operations().insert_dataframe_to_mongodb(df_for_storage)

    



    





