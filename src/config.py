"""
配置模块
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """系统配置"""

    # Ollama配置
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

    # 连接设置
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "30"))
    ollama_max_retries: int = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))

    # 向量数据库配置
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/vector_store")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # HuggingFace配置
    hf_cache_folder: Optional[str] = os.getenv("HF_CACHE_FOLDER", "./.hf_cache")

    # RAG配置
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    max_context_length: int = int(os.getenv("MAX_CONTEXT_LENGTH", "3000"))

    # 系统配置
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def ollama_api_url(self) -> str:
        """获取Ollama API完整URL"""
        return f"{self.ollama_base_url}/api"


config = Config()
