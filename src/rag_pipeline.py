"""
RAG管道核心模块
"""

import chromadb
from chromadb.config import Settings
from typing import Any, Dict, List, Optional
import dspy
import numpy as np

from src.config import config
from src.document_loader import Document
from src.retrievers import ChromaDBRetriever
from src.language_models import RemoteOllamaLM
from src.embeddings import LocalEmbeddings
import logging

logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG管道主类"""

    def __init__(self):
        # 初始化嵌入模型
        self.embeddings = LocalEmbeddings()

        # 初始化ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=config.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # 初始化自定义ChromaDB检索器
        self.retriever = ChromaDBRetriever(
            collection_name="documents",
            persist_directory=config.chroma_persist_dir,
            embedding_function=self.embeddings.embed,
            k=config.retrieval_top_k,
        )

        # 初始化Ollama语言模型
        # self.lm = OllamaLM()
        self.lm = RemoteOllamaLM(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            timeout=config.ollama_timeout,
        )

        # 配置DSPy
        dspy.configure(lm=self.lm, rm=self.retriever)

        # 定义DSPy签名
        self.define_signatures()

    def define_signatures(self):
        """定义DSPy签名"""

        class GenerateAnswer(dspy.Signature):
            """基于上下文回答问题"""

            context = dspy.InputField(desc="相关文档内容")
            question = dspy.InputField(desc="用户问题")
            answer = dspy.OutputField(desc="简洁准确的答案", format=lambda x: str(x))

        class GenerateAnswerWithReasoning(dspy.Signature):
            """带有推理过程的回答"""

            context = dspy.InputField(desc="相关文档内容")
            question = dspy.InputField(desc="用户问题")
            reasoning = dspy.OutputField(desc="思考过程")
            answer = dspy.OutputField(desc="最终答案")

        self.GenerateAnswer = GenerateAnswer
        self.GenerateAnswerWithReasoning = GenerateAnswerWithReasoning

        # 创建预测模块
        self.answer_generator = dspy.ChainOfThought(GenerateAnswer)
        self.reasoning_generator = dspy.ChainOfThought(GenerateAnswerWithReasoning)

    def index_documents(
        self, documents: List[Document], collection_name: str = "documents"
    ):
        """索引文档到向量数据库"""

        # 创建或获取集合
        collection = self.chroma_client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        # 准备数据
        ids = []
        texts = []
        metadatas = []

        for doc in documents:
            ids.append(doc.id)
            texts.append(doc.content)
            metadatas.append(
                {**doc.metadata, "source": doc.source, "chunk_index": doc.chunk_index}
            )

        # 生成嵌入
        print("生成文档嵌入...")
        embeddings = np.asarray(self.embeddings.embed(texts), dtype=np.float32)

        # 添加到集合
        collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts
        )

        print(f"✅ 已索引 {len(documents)} 个文档块")
        return collection.count()

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """检索相关文档"""
        k = k or config.retrieval_top_k

        # 使用自定义检索器
        passages = self.retriever(query, k=k)

        results = []
        for i, doc in enumerate(passages):
            results.append(
                {
                    "rank": i + 1,
                    "content": doc,
                    "score": 1.0 - (i * 0.1),  # 简单评分
                    "source": "chromadb",
                }
            )

        return results

    def answer_question(
        self, question: str, use_reasoning: bool = False
    ) -> Dict[str, Any]:
        """回答问题"""
        logger.info(f"收到问题----------: {question}")
        # 检索相关文档
        retrieved_docs = self.retrieve(question)
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])

        # 截断上下文以避免超出长度限制
        if len(context) > config.max_context_length:
            context = context[: config.max_context_length] + "..."

        logger.info(f"检索到的上下文: {context[:200]}...")

        # 生成答案
        logger.info(f"开始生成答案, use_reasoning={use_reasoning}")
        if use_reasoning:
            logger.info("使用reasoning_generator")
            pred = self.reasoning_generator(context=context, question=question)
            answer = pred.answer
            reasoning = pred.reasoning
            logger.info(f"Pred对象: {pred}")
            logger.info(f"Answer: {answer}, Reasoning: {reasoning}")
        else:
            logger.info("使用answer_generator")
            pred = self.answer_generator(context=context, question=question)
            answer = pred.answer
            reasoning = None
            logger.info(f"Pred对象: {pred}")
            logger.info(f"Answer: {answer}")

        logger.info(f"生成的答案: {answer}")

        return {
            "question": question,
            "answer": answer,
            "reasoning": reasoning,
            "sources": retrieved_docs,
            "context_used": context[:500] + "..." if len(context) > 500 else context,
        }

    def interactive_session(self):
        """交互式问答会话"""
        print("\n" + "=" * 60)
        print("🤖 本地RAG助手 (输入 'quit' 或 'exit' 退出)")
        print("=" * 60)

        while True:
            try:
                question = input("\n❓ 你的问题: ").strip()

                if question.lower() in ["quit", "exit", "q"]:
                    print("👋 再见!")
                    break

                if not question:
                    continue

                print("🧠 思考中...")
                result = self.answer_question(question, use_reasoning=True)

                print(f"\n📝 回答: {result['answer']}")

                if result["reasoning"]:
                    print(f"\n💭 推理过程: {result['reasoning']}")

                print(f"\n📚 参考文档:")
                for i, source in enumerate(result["sources"][:3], 1):
                    print(f"  {i}. {source['content'][:200]}...")

            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
