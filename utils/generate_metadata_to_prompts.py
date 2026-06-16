import os
import re
import pandas as pd

class Cks_to_Markdown:

    """
    chunks key: docnm_kwd, content_with_weight, img_path
    """
    metadata_keys = ['source_filename', 'intro_content', 'img_url']
    minio_address = os.getenv("MINIO_URL")
    raw_pdf_file_lst = ['66001-00935-SSP_V1.3', 'GD5000？07版本软件升级指导_A0']
    video_filename = '知识库视频介绍.xlsx' # 检测视频xlsx介绍文件，当检测到时，启用视频方式生产metadata

    @classmethod
    def xlsx_metadata(cls, chunk: dict):
        df = pd.DataFrame(columns=cls.metadata_keys)
        # 检测docnm_kwd的后缀名，分别进行处理
        df.loc[0, 'source_filename'] = re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1].replace("_修正版.xlsx",
                                                                                              ".xlsx").replace(
            "_附加表.xlsx", ".docx")

        # 检测docnm_kwd的后缀名，针对pdf切分出来的xlsx处理，需要对原生pdf与word转pdf分别处理
        if re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1].endswith('_enhanced_images.xlsx'):
            if re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1].split('.')[0] in cls.raw_pdf_file_lst:
                df.loc[0, 'source_filename'] = re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1].split('.')[0] + '.pdf'
            else:
                df.loc[0, 'source_filename'] = re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1].split('.')[0] + '.docx'

        df.loc[0, 'intro_content'] = chunk['content_with_weight'].replace("\n", "")
        df.loc[0, 'img_url'] = ""
        # print(df)
        return df

    @classmethod
    def video_metadata(cls, chunk: dict):
        df = pd.DataFrame(columns=cls.metadata_keys)
        # 检测docnm_kwd的后缀名，分别进行处理
        df.loc[0, 'source_filename'] = re.search(r'视频名称:\s*(.*?)(?=\s*;|$)', chunk['content_with_weight']).group(
            1) + '.mp4'
        df.loc[0, 'intro_content'] = chunk['content_with_weight'].replace("\n", "")
        df.loc[0, 'img_url'] = cls.minio_address + "/" + chunk['image_id']
        # print(df)
        return df

    @classmethod
    def docx_metadata(cls, chunk: dict):
        df = pd.DataFrame(columns=cls.metadata_keys)
        df.loc[0, 'source_filename'] = re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1].replace("_主文档", "")
        df.loc[0, 'intro_content'] = chunk['content_with_weight'].replace("\n", "")
        try:
            if chunk['image_id'] == "":
                df.loc[0, 'img_url'] = chunk['image_id']
            else:
                df.loc[0, 'img_url'] = cls.minio_address + "/" + chunk['image_id']
        except:
            df.loc[0, 'img_url'] = ""
        # print(df)
        return df

    @classmethod
    def pdf_metadata(cls, chunk: dict):
        df = pd.DataFrame(columns=cls.metadata_keys)
        df.loc[0, 'source_filename'] = re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1].replace("_主文档", "")
        df.loc[0, 'intro_content'] = chunk['content_with_weight'].replace("\n", "")
        try:
            if chunk['image_id'] == "":
                df.loc[0, 'img_url'] = chunk['image_id']
            else:
                df.loc[0, 'img_url'] = cls.minio_address + "/" + chunk['image_id']
        except:
            df.loc[0, 'img_url'] = ""
        # print(df)
        return df

    @classmethod
    def markdown_metadata(cls, chunk: dict):
        df = pd.DataFrame(columns=cls.metadata_keys)
        # 需要对原生pdf与word转pdf分别处理
        if re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1].split('.')[0] in cls.raw_pdf_file_lst:
            df.loc[0, 'source_filename'] = re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1].split('.')[0] + '.pdf'
        else:
            df.loc[0, 'source_filename'] = re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1].split('.')[0] + '.docx'

        df.loc[0, 'intro_content'] = chunk['content_with_weight'].replace("\n", "")
        try:
            if chunk['image_id'] == "":
                df.loc[0, 'img_url'] = chunk['image_id']
            else:
                df.loc[0, 'img_url'] = cls.minio_address + "/" + chunk['image_id']
        except:
            df.loc[0, 'img_url'] = ""
        # print(df)
        return df

    @classmethod
    def summary_template_by_markdown(cls, doc_name, img_id, page_content):
        # markdown文本模板
        md_template = f"- 文档来源: {doc_name}\n"
        if img_id != "":  # 动态检测图片并嵌入
            if ".mp4" in img_id:
                md_template += f"- 相关视频: http://{img_id}\n"
            else:
                md_template += f"- 相关插图: http://{img_id}\n"
        md_template += f"- 参考内容:\n{page_content}\n\n"
        # print(md_template)
        return md_template

    @classmethod
    def metadata_to_prompts(cls, raw_chunks: list) -> str:
        """
        chunks key: docnm_kwd, content_with_weight, img_path
        """
        metadata_keys = ['source_filename', 'intro_content', 'img_url']
        df = pd.DataFrame(columns=metadata_keys)
        for chunk in raw_chunks:
            # print(chunk)
            df_temp = pd.DataFrame(columns=metadata_keys)
            # 检查chunk中 docnm_kwd 文件类型，根据类型不同采用不同的方法
            if re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1].split('.')[-1] == 'xlsx':
                if re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1] == cls.video_filename:
                    df_temp = cls.video_metadata(chunk)
                else:
                    df_temp = cls.xlsx_metadata(chunk)
            elif re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1].split('.')[-1] == 'docx':
                df_temp = cls.docx_metadata(chunk)
            elif re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1].split('.')[-1] == 'pdf':
                df_temp = cls.pdf_metadata(chunk)
            elif re.split(r'\\|\\\\|/', chunk['docnm_kwd'])[-1].split('.')[-1] == 'md':
                df_temp = cls.markdown_metadata(chunk)

            df = pd.concat([df, df_temp], axis=0).reset_index(drop=True)
        # print(df)
        # markdown_table = df.to_markdown(index=False)
        # # print(markdown_table)
        
        result_md = ""
        for i in range(df.shape[0]):
            result_md += cls.summary_template_by_markdown(df.loc[i, 'source_filename'], df.loc[i, 'img_url'], df.loc[i, 'intro_content'])
        # print(result_md)
        return result_md





