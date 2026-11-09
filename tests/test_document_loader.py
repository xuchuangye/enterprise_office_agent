"""
测试文档读取加载和分块
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

from knowledge.document_loader import UniversalDocumentLoader

# 创建一个文档加载器实例
loader = UniversalDocumentLoader()

# 加载文档
loaded_docs = loader.load_file("data/knowledge_base/白泽团队全面介绍.docx")

print(f"加载了 {len(loaded_docs)} 个文档")
for doc in loaded_docs:
    # 文档分块是按照
    print(f"Content: {doc.page_content[:100]}...")  # 打印前100个字符
    print(f"Metadata: {doc.metadata}")
    print("-" * 20)