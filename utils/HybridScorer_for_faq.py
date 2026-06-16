import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVectorParams
from typing import Tuple, Dict


class HybridScorer:
    def __init__(self, dense_model_name: str = "thenlper/gte-large",
                 sparse_model_name: str = "Qdrant/minicoil-v1"):
        """
        初始化双编码器模型
        :param dense_model_name: 稠密向量模型路径/名称
        :param sparse_model_name: 稀疏向量模型路径/名称
        """
        # 加载稠密语义编码模型（捕捉深层语义）
        self.dense_encoder = SentenceTransformer(dense_model_name)
        # 连接稀疏向量服务（关键词增强）
        self.sparse_client = QdrantClient()
        self.sparse_params = SparseVectorParams(model=sparse_model_name)

    def _get_dense_score(self, query: str, reference: str) -> float:
        """
        计算稠密向量余弦相似度
        :param query: 用户问题
        :param reference: 参考问题
        :return: 语义相似度得分 [0,1]
        """
        # 生成1024维语义向量
        query_embed = self.dense_encoder.encode(query)
        ref_embed = self.dense_encoder.encode(reference)
        # 余弦相似度计算
        return np.dot(query_embed, ref_embed) / (np.linalg.norm(query_embed) * np.linalg.norm(ref_embed))

    def _get_sparse_score(self, query: str, reference: str) -> float:
        """
        计算稀疏向量点积相似度（基于miniCOIL）
        :return: 关键词匹配得分 [0,∞)
        """
        # 生成稀疏向量（仅存储非零值）
        sparse_query = self.sparse_client.encode(query, self.sparse_params)
        sparse_ref = self.sparse_client.encode(reference, self.sparse_params)

        # 高效计算点积（仅处理共同非零维度）
        score = 0.0
        for idx, weight in sparse_query.items():
            if idx in sparse_ref:
                score += weight * sparse_ref[idx]
        return score

    def hybrid_score(self, query: str, reference: str,
                     alpha: float = 0.4,
                     normalize_sparse: bool = True) -> Tuple[float, Dict]:
        """
        混合加权评分核心方法
        :param alpha: 稀疏向量权重系数（稠密权重=1-alpha）
        :param normalize_sparse: 是否对稀疏得分做sigmoid归一化
        :return: (加权总分, 各分项得分详情)
        """
        # 并行计算两种得分
        dense_score = self._get_dense_score(query, reference)
        sparse_score = self._get_sparse_score(query, reference)

        # 稀疏得分归一化（避免量纲差异）
        processed_sparse = 1 / (1 + np.exp(-sparse_score)) if normalize_sparse else sparse_score

        # 线性加权融合
        combined_score = (1 - alpha) * dense_score + alpha * processed_sparse

        return combined_score, {
            "dense_score": dense_score,
            "sparse_score": sparse_score,
            "normalized_sparse": processed_sparse,
            "alpha": alpha
        }