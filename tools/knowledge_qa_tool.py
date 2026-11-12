"""
知识问答工具
基于RAG的企业知识库问答
"""
from typing import Iterator, Optional, Type, ClassVar
from pydantic import BaseModel, Field, PrivateAttr
import logging

from config.settings import settings
from .base import BaseTool
from knowledge import KnowledgeRetriever
from knowledge.embeddings import get_embeddings
# 推荐使用 langchain_huggingface，但为避免额外依赖，先继续用社区版，仅修复 deprecation 警告
from langchain_community.llms import Tongyi
# from langchain.llms.base import BaseLLM
from langchain_core.language_models.llms import BaseLLM

logger = logging.getLogger(__name__)


class KnowledgeQAInput(BaseModel):
    """知识问答工具输入参数模式"""
    question: str = Field(description="要询问的问题")
    top_k: Optional[int] = Field(4, description="检索相关文档数量")


class KnowledgeQATool(BaseTool):
    """
    企业知识库问答工具
    
    功能：
    基于内部知识库（规章制度、产品文档等）回答问题，采用RAG模式：
    1. 从向量数据库检索相关内容
    2. 结合LLM生成准确答案
    """
    
    name: str = "knowledge_qa"
    description: str = """
    企业知识库问答工具，用于回答企业内部制度、产品知识、流程规范等问题。
    输入您的问题，工具会从知识库中检索相关信息并给出答案。
    示例问题：
    - 公司的年假政策是什么？
    - 如何申请出差报销？
    - 产品的主要功能有哪些？
    """
    args_schema: Type[BaseModel] = KnowledgeQAInput  # type: ignore
    # 答案由 LLM 生成，支持逐 token 流式返回
    streams_llm_output: ClassVar[bool] = True

    # 使用 PrivateAttr 定义“私有”属性，绕过 Pydantic 验证
    _retriever: KnowledgeRetriever = PrivateAttr()
    _llm: Optional[BaseLLM] = PrivateAttr(default=None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 使用默认集合名称，确保与前端保持一致
        # 集合名称类似于数据库的表名
        self._retriever = KnowledgeRetriever(collection_name="enterprise_knowledge")
        self._llm = None
        self._init_llm()
    
    def _init_llm(self):
        """初始化LLM"""
        try:
            if settings.DASHSCOPE_API_KEY:
                import os
                os.environ["DASHSCOPE_API_KEY"] = settings.DASHSCOPE_API_KEY
                self._llm = Tongyi(model_name=settings.LLM_MODEL_NAME)
                logger.info("通义千问LLM初始化成功")
            else:
                logger.warning("未配置DASHSCOPE_API_KEY，将使用检索内容直接返回")
                self._llm = None
        except Exception as e:
            logger.error(f"LLM初始化失败: {e}")
            self._llm = None
    
    def _retrieve(self, question: str, top_k: int):
        """检索相关文档并构建上下文"""
        docs = self._retriever.retrieve(question, top_k=top_k)
        context_parts = []
        sources = set()
        for i, doc in enumerate(docs or [], 1):
            source = doc.metadata.get("source", "未知来源")
            if "sheet_name" in doc.metadata:
                source += f" (工作表: {doc.metadata['sheet_name']})"
            sources.add(source)
            context_parts.append(f"[参考资料{i}]\n{doc.page_content}")
        return docs or [], sources, "\n\n".join(context_parts)

    @staticmethod
    def _build_prompt(context: str, question: str) -> str:
        return f"""你是一个企业智能办公助手，请根据以下参考资料回答用户的问题。
        如果参考资料无法回答问题，请如实说明，不要编造信息。

        参考资料：
        {context}

        用户问题：{question}

        请用专业、清晰的语言回答，并在回答末尾注明参考来源。
        """

    @staticmethod
    def _fallback_answer(docs, sources) -> str:
        """无LLM时直接返回检索内容摘要"""
        answer = "基于知识库检索到以下相关信息（请自行判断准确性）：\n\n"
        for i, doc in enumerate(docs, 1):
            answer += f"{i}. {doc.page_content[:200]}...\n"
        answer += f"\n📚 参考来源：{', '.join(sources)}"
        return answer

    def _execute(self, question: str, top_k: int = 4, **kwargs) -> str:
        """执行知识问答"""
        try:
            # 1. 检索相关文档
            docs, sources, context = self._retrieve(question, top_k)

            if not docs:
                return "抱歉，在知识库中未找到与您问题相关的信息。请联系管理员补充相关知识。"

            # 2. 如果有LLM，使用RAG生成答案
            if self._llm:
                response = self._llm.invoke(self._build_prompt(context, question))
                answer = response.strip()

                # 添加来源信息（如果LLM未包含）
                if "来源" not in answer and "参考资料" not in answer:
                    answer += f"\n\n📚 参考来源：{', '.join(sources)}"
            else:
                answer = self._fallback_answer(docs, sources)

            return answer

        except Exception as e:
            logger.error(f"知识问答执行失败: {e}")
            return f"问答处理出错: {str(e)}"

    def stream_result(self, question: str, top_k: int = 4, **kwargs) -> Iterator[str]:
        """流式执行知识问答：答案边生成边返回，无需等整段结束"""
        try:
            docs, sources, context = self._retrieve(question, top_k)

            if not docs:
                yield "抱歉，在知识库中未找到与您问题相关的信息。请联系管理员补充相关知识。"
                return

            if not self._llm:
                yield self._fallback_answer(docs, sources)
                return

            # Tongyi 实现了 _stream，llm.stream 会逐 token 产出
            answer_parts = []
            for chunk in self._llm.stream(self._build_prompt(context, question)):
                text = chunk if isinstance(chunk, str) else str(chunk)
                if text:
                    answer_parts.append(text)
                    yield text

            answer = "".join(answer_parts).strip()
            if "来源" not in answer and "参考资料" not in answer:
                yield f"\n\n📚 参考来源：{', '.join(sources)}"

        except Exception as e:
            logger.error(f"知识问答流式输出失败: {e}")
            yield f"问答处理出错: {str(e)}"