"""
嵌入模型模块
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer

from src.config import config


class LocalEmbeddings:
    """本地嵌入模型"""

    # 类级别的模型缓存
    _model_cache: Dict[str, SentenceTransformer] = {}

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.embedding_model

        # 创建HF缓存目录（如果配置了）
        if config.hf_cache_folder:
            Path(config.hf_cache_folder).mkdir(parents=True, exist_ok=True)

        # 检查缓存中是否已有该模型
        if self.model_name in LocalEmbeddings._model_cache:
            print(f"✅ 使用缓存的嵌入模型: {self.model_name}")
            self.model = LocalEmbeddings._model_cache[self.model_name]
        else:
            print(f"📥 加载嵌入模型: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name, cache_folder=config.hf_cache_folder
            )
            # 缓存模型实例
            LocalEmbeddings._model_cache[self.model_name] = self.model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入"""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
