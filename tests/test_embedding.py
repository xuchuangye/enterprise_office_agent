"""
测试向量数据库的功能，包括添加文档、相似度检索和集合管理等功能，
确保向量数据库能够正确地存储和检索知识库中的文档，并且支持持久化机制，能够在程序重启后继续使用之前的数据。
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

from knowledge.embeddings import get_embeddings

# 获取Embedding模型
embeddings = get_embeddings()

# 测试文本向量化
text = "零基础，学AI，找白泽，不踩坑"
# 向量化查询文本，得到一个文本的向量表示
vector = embeddings.embed_query(text)

print(f"文本: {text}")
print(f"向量维度: {len(vector)}, 向量示例: {vector[:5]}")
print(f"Vector: {vector[:10]}")

