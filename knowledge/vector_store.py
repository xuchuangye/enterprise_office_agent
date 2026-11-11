"""
向量数据库管理模块
"""
import os
import logging
from typing import List, Optional
from langchain_chroma import Chroma # 为什么使用Chroma？轻量级，非常适合RAG应用，开源，免费，易于使用，支持持久化，与LangChain无缝集成
from langchain_milvus import Milvus
from langchain_pinecone import Pinecone
from langchain_core.documents import Document

from config.settings import settings
from .embeddings import get_embeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    向量数据库管理：
    1. 初始化向量数据库以及重试
    2. 添加文档到向量数据库
    3. 相似度检索（带分数和不带分数的）
    """
    def __init__(self, collection_name: str = "enterprise_knowledge"):
        # 定义集合名称，类似于数据库的表名，确保与前端保持一致
        self.collection_name = collection_name
        # 持久化目录，数据存储的位置，数据保存到本地磁盘
        # 持久化机制：会自动将数据保存到持久化目录当中，持久化意味着数据不会因为程序重启而丢失
        # 可以随时加载已经有的知识库，而且支持增量更新知识库
        self.persist_directory = os.path.join(settings.CHROMA_PERSIST_DIR, collection_name)
        os.makedirs(self.persist_directory, exist_ok=True)

        # 获取嵌入模型
        self.embeddings = get_embeddings()
        self.vectorstore = None
        self._init_vectorstore()

    def _init_vectorstore(self):
        """安全初始化 Chroma 向量存储"""
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,    # 集合名称
                embedding_function=self.embeddings,      # 嵌入模型
                persist_directory=self.persist_directory # 持久化目录
            )
            logger.info(f"向量数据库初始化成功，集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {e}", exc_info=True)
            self.vectorstore = None
            # 你可以在此处提供更友好的提示，例如：
            # print("向量数据库不可用，请联系管理员")

    def ensure_ready(self):
        """如果之前初始化失败，可以重新尝试一次"""
        if self.vectorstore is None:
            self._init_vectorstore()
        return self.vectorstore is not None

    # 添加文档到向量数据库
    def add_documents(self, documents: List[Document]) -> bool:
        if not self.ensure_ready():
            logger.error("向量数据库未初始化，无法添加文档")
            return False
        try:
            # 增量更新数据库（追加式增量更新，不是严格意义上的文档级增量更新）
            self.vectorstore.add_documents(documents) # type: ignore
            # Chroma 新版本不再需要手动 persist，自动持久化
            logger.info(f"成功添加 {len(documents)} 个文档片段")
            return True
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False
    
    # 相似度检索，从向量数据库中检索相关文档，返回相似度最匹配的前 k 个文档
    # 这里应用了Chroma的关键特性，添加文档后会自动持久化，保存到本地磁盘
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if not self.ensure_ready():
            return []
        try:
            # 相似度检索，是基于向量的距离找到最相关的文档，而且支持添加元数据
            return self.vectorstore.similarity_search(query, k=k) # type: ignore
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []

    # 相似度检索，返回文档和相似度分数
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        if not self.ensure_ready():
            return []
        try:
            return self.vectorstore.similarity_search_with_score(query, k=k) # type: ignore
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []

    # 集合的管理：支持创建、删除、清空
    def delete_collection(self) -> bool:
        if not self.ensure_ready():
            return False
        try:
            self.vectorstore.delete_collection() # type: ignore
            logger.info(f"集合 {self.collection_name} 已删除")
            # 删除集合后将vectorstore设置为None，以便下次使用时重新初始化
            self.vectorstore = None
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False

    # 获取集合中的文档数量
    def get_document_count(self) -> int:
        if not self.ensure_ready():
            return 0
        try:
            collection_data = self.vectorstore.get() # type: ignore
            return len(collection_data.get("documents", []))
        except Exception:
            return 0
    # 将向量数据库作为检索器
    def as_retriever(self, search_kwargs: dict):
        if not self.ensure_ready():
            return None
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs or {"k": 4}) # type: ignore