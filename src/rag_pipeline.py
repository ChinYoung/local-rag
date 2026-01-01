"""
RAG管道核心模块
"""

import chromadb
from chromadb.config import Settings
from typing import Any, Dict, List, Optional, cast
import dspy
import ollama
from sentence_transformers import SentenceTransformer
import numpy as np

from src.config import config
from src.document_loader import Document, DocumentLoader


class ChromaDBRetriever(dspy.Retrieve):
    """Custom ChromaDB Retriever for DSPy"""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None,
        k: int = 3,
    ):
        super().__init__(k=k)
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function

        # Initialize ChromaDB client
        if persist_directory:
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self.chroma_client = chromadb.EphemeralClient()
        self.collection = None

    def _get_collection(self):
        """Get or create collection"""
        if self.collection is None:
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
        return self.collection

    def forward(self, query: str, k: Optional[int] = None, **kwargs) -> List[str]:
        """Retrieve documents from ChromaDB"""
        k = k or self.k
        collection = self._get_collection()

        # Generate embedding for query
        if self.embedding_function is None:
            raise ValueError("embedding_function must be provided")
        query_embedding = self.embedding_function([query])[0]

        # Query collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        passages = []
        if results and results["documents"]:
            for doc_list in results["documents"]:
                passages.extend(doc_list)

        return passages


class OllamaLM(dspy.LM):
    """Ollama语言模型适配器"""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        selected_model = model or config.ollama_model
        super().__init__(selected_model)
        self.model = selected_model
        self.base_url = base_url or config.ollama_base_url

        # 测试连接
        try:
            response = ollama.list()
            print(
                f"✅ 连接到 Ollama，可用模型: {[m['name'] for m in response['models']]}"
            )
        except Exception as e:
            raise ConnectionError(f"无法连接到 Ollama ({self.base_url}): {e}")

    def basic_request(self, prompt: str, **kwargs):
        """基础请求方法"""
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
                "top_p": kwargs.get("top_p", 0.9),
            },
        )
        return response

    def __call__(
        self,
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        response = self.basic_request(
            prompt or "",
            max_tokens=max_tokens,
            **kwargs,
        )
        return [response["response"]]


class LocalEmbeddings:
    """本地嵌入模型"""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.embedding_model
        print(f"加载嵌入模型: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入"""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()


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
        self.lm = OllamaLM()

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

        # 检索相关文档
        retrieved_docs = self.retrieve(question)
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])

        # 截断上下文以避免超出长度限制
        if len(context) > config.max_context_length:
            context = context[: config.max_context_length] + "..."

        # 生成答案
        if use_reasoning:
            pred = self.reasoning_generator(context=context, question=question)
            answer = pred.answer
            reasoning = pred.reasoning
        else:
            pred = self.answer_generator(context=context, question=question)
            answer = pred.answer
            reasoning = None

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


class RAGOptimizer:
    """RAG优化器"""

    @staticmethod
    def optimize_with_bootstrap(rag_pipeline, train_examples):
        """使用BootstrapFewShot优化"""

        class RAG(dspy.Module):
            def __init__(self):
                super().__init__()
                self.retrieve = dspy.Retrieve(k=3)
                self.generate_answer = dspy.ChainOfThought(rag_pipeline.GenerateAnswer)

            def forward(self, question):
                passages = cast(List[str], self.retrieve(question))
                context = "\n\n".join(passages) if passages else ""
                return self.generate_answer(context=context, question=question)

        # 定义评估指标
        def validate_answer(example, pred, trace=None):
            # 简单评估：检查答案是否包含关键词
            gold_answer = example.answer.lower()
            pred_answer = pred.answer.lower()

            # 计算重叠词的比例
            gold_words = set(gold_answer.split())
            pred_words = set(pred_answer.split())

            if not gold_words:
                return 0

            overlap = len(gold_words.intersection(pred_words)) / len(gold_words)
            return overlap > 0.5  # 至少50%重叠

        # 优化
        teleprompter = dspy.BootstrapFewShot(metric=validate_answer)
        optimized_rag = teleprompter.compile(RAG(), trainset=train_examples)

        return optimized_rag
