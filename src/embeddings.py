"""
嵌入模型模块
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer

from src.config import config


class LocalEmbeddings:
    """本地嵌入模型"""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.embedding_model
        print(f"加载嵌入模型: {self.model_name}")
        self.model = SentenceTransformer(
            self.model_name, cache_folder=config.hf_cache_folder
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入"""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
