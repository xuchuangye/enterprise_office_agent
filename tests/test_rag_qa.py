"""
测试向量检索
"""
import sys
from pathlib import Path
# 1. 获取当前文件(test_agent.py)的绝对路径
current_file = Path(__file__).resolve()
# 2. 向上回退两级目录: tests -> enterprise_office_agent (项目根目录)
project_root = current_file.parent.parent
# 3. 将项目根目录插入到系统路径的最前面(索引0)
# 这确保了Python会优先在你的项目中查找模块，而不是去site-packages里找
sys.path.insert(0, str(project_root))

from knowledge.vector_store import VectorStoreManager
from langchain_core.documents import Document

# 创建向量数据库管理器
# vectorstore_manager = VectorStoreManager()
vectorstore_manager = VectorStoreManager(collection_name="test_rag_qa_001")

# 添加文档
documents = [
    Document(page_content="这是一个测试文档", metadata={"source": "test_001", "page": 1}),
    Document(page_content="企业智能办公助手系统是开发的一款基于大模型的企业办公辅助工具", metadata={"source": "test_002", "page": 1}),
]

# 在向量数据库中添加文档
vectorstore_manager.add_documents(documents)

# 测试检索
results = vectorstore_manager.similarity_search("企业智能助手", k=3)

for result in results:
    print(f"Content: {result.page_content}, Metadata: {result.metadata}")
    print(f"Source: {result.metadata.get('source')}, Page: {result.metadata.get('page')}")
    print("-" * 20)

