"""
测试文件
"""

import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.document_loader import DocumentLoader, Document
from src.rag_pipeline import LocalEmbeddings


def test_document_loader():
    """测试文档加载器"""
    # 创建测试文档
    test_content = "这是一个测试文档。\n" * 10
    doc = Document(
        id="test_1", content=test_content, metadata={"test": True}, source="test.txt"
    )

    # 测试分块
    chunks = DocumentLoader.chunk_document(doc, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)

    # 验证内容
    reconstructed = " ".join([chunk.content for chunk in chunks])
    assert len(reconstructed) >= len(test_content) * 0.8  # 允许少量损失


def test_embeddings():
    """测试嵌入模型"""
    embeddings = LocalEmbeddings()
    texts = ["测试文本1", "测试文本2"]

    vectors = embeddings.embed(texts)

    assert len(vectors) == len(texts)
    assert all(len(vec) == 384 for vec in vectors)  # MiniLM向量维度


@pytest.mark.skipif(not Path("./data/vector_store").exists(), reason="需要先运行索引")
def test_rag_pipeline():
    """测试RAG管道"""
    from src.rag_pipeline import RAGPipeline

    rag = RAGPipeline()

    # 测试检索
    results = rag.retrieve("测试查询", k=2)
    assert isinstance(results, list)

    # 测试问答（如果有数据）
    if len(results) > 0:
        answer = rag.answer_question("测试问题")
        assert "answer" in answer
        assert "sources" in answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
