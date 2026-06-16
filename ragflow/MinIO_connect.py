import logging
import os
import time

from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import json

load_dotenv()
minio_config = {
    "host": os.getenv("MINIO_URL"),
    "user": os.getenv("MINIO_ACCESS_KEY"),
    "password": os.getenv("MINIO_SECRET_KEY"),
    }

minio_bucket_main = os.getenv("BUCKET_FOR_RAG_IMAGE") # 存储知识库的图片
minio_bucket_temporary_chat = os.getenv("BUCKET_FOR_TEMP_IMAGE") # 存储用户生成的图片

class Minio_Connection:
    def __init__(self):
        self.bucket_main = minio_bucket_main
        self.conn = Minio(minio_config["host"],
                              access_key=minio_config["user"],
                              secret_key=minio_config["password"],
                              secure=False
                              )

    def __open__(self):
        try:
            if self.conn:
                self.__close__()
        except Exception:
            pass

        try:
            self.conn = Minio(minio_config["host"],
                              access_key=minio_config["user"],
                              secret_key=minio_config["password"],
                              secure=False
                              )
        except Exception:
            logging.exception(
                "Fail to connect %s " % minio_config["host"])

    def __close__(self):
        del self.conn
        self.conn = None

    def health(self):
        bucket, fnm, binary = "txtxtxtxt1", "txtxtxtxt1", b"_t@@@1"
        if not self.conn.bucket_exists(bucket):
            self.conn.make_bucket(bucket)
        r = self.conn.put_object(bucket, fnm,
                                 BytesIO(binary),
                                 len(binary)
                                 )
        return r
    
    def create_bucket(self, bucket_name):
        """创建存储桶（如果不存在）"""
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' has been created successfully!")
        else:
            print(f"Bucket '{bucket_name}' already exists!")

    def put(self, bucket, fnm, binary):
        for _ in range(3):
            try:
                if not self.conn.bucket_exists(bucket):
                    self.conn.make_bucket(bucket)

                r = self.conn.put_object(bucket, fnm,
                                         BytesIO(binary),
                                         len(binary),
                                         content_type="image/png"
                                         )
                return r
            except Exception:
                logging.exception(f"Fail to put {bucket}/{fnm}:")
                self.__open__()
                time.sleep(1)

    def rm(self, bucket, fnm):
        try:
            self.conn.remove_object(bucket, fnm)
        except Exception:
            logging.exception(f"Fail to remove {bucket}/{fnm}:")

    def get(self, bucket, filename):
        for _ in range(1):
            try:
                r = self.conn.get_object(bucket, filename)
                return r
                # return r.read()
            except Exception:
                logging.exception(f"Fail to get {bucket}/{filename}")
                self.__open__()
                time.sleep(1)
        return

    def obj_exist(self, bucket, filename):
        try:
            if not self.conn.bucket_exists(bucket):
                return False
            if self.conn.stat_object(bucket, filename):
                return True
            else:
                return False
        except S3Error as e:
            if e.code in ["NoSuchKey", "NoSuchBucket", "ResourceNotFound"]:
                return False
        except Exception:
            logging.exception(f"obj_exist {bucket}/{filename} got exception")
            return False

    def get_presigned_url(self, bucket, fnm, expires):
        for _ in range(10):
            try:
                return self.conn.get_presigned_url("GET", bucket, fnm, expires)
            except Exception:
                logging.exception(f"Fail to get_presigned {bucket}/{fnm}:")
                self.__open__()
                time.sleep(1)
        return

    def remove_bucket(self, bucket):
        try:
            if self.conn.bucket_exists(bucket):
                objects_to_delete = self.conn.list_objects(bucket, recursive=True)
                for obj in objects_to_delete:
                    self.conn.remove_object(bucket, obj.object_name)
                self.conn.remove_bucket(bucket)
        except Exception:
            logging.exception(f"Fail to remove bucket {bucket}")

    def change_bucket_access_to_open(self, bucket, filepath):
        # 定义策略：允许匿名用户读取桶内所有文件
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                # 存储桶级操作
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:ListBucket"],  # 桶级操作
                    "Resource": [f"arn:aws:s3:::{bucket}"],  # 必须是桶ARN
                    "Condition": {"StringLike": {"s3:prefix": [f"{filepath}/*"]}}  # 可选：限制目录
                },
                # 对象级操作
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],  # 对象级操作
                    "Resource": [f"arn:aws:s3:::{bucket}/{filepath}/*"]  # 对象路径
                }
            ]
        }
        
        self.conn.set_bucket_policy(bucket, json.dumps(policy))

if __name__ == '__main__':
    bkt = "invt-agent-images"
    img_path = "images"
    Minio_Connection().change_bucket_access_to_open(bucket=bkt, filepath=img_path)
    

