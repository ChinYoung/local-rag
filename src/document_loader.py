"""
文档加载和预处理模块
"""

import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import PyPDF2
import docx2txt
import markdown
from tqdm import tqdm


@dataclass
class Document:
    """文档数据类"""

    id: str
    content: str
    metadata: Dict[str, Any]
    source: str
    chunk_index: int = 0


class DocumentLoader:
    """文档加载器"""

    @staticmethod
    def generate_doc_id(content: str, source: Union[str, Path]) -> str:
        """生成文档ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{Path(source).stem}_{content_hash}"

    @staticmethod
    def load_pdf(file_path: Union[str, Path]) -> str:
        """加载PDF文件"""
        content = []
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    content.append(f"Page {page_num}:\n{text}")
        return "\n\n".join(content)

    @staticmethod
    def load_txt(file_path: Union[str, Path]) -> str:
        """加载文本文件"""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def load_docx(file_path: Union[str, Path]) -> str:
        """加载DOCX文件"""
        return docx2txt.process(file_path)

    @staticmethod
    def load_markdown(file_path: Union[str, Path]) -> str:
        """加载Markdown文件"""
        with open(file_path, "r", encoding="utf-8") as file:
            md_content = file.read()
            # 可选：转换为纯文本或保留标记
            return md_content

    @classmethod
    def load_document(cls, file_path: Union[str, Path]) -> Optional[Document]:
        """加载单个文档"""
        file_path = Path(file_path)
        if not file_path.exists():
            return None

        try:
            if file_path.suffix.lower() == ".pdf":
                content = cls.load_pdf(str(file_path))
            elif file_path.suffix.lower() == ".txt":
                content = cls.load_txt(str(file_path))
            elif file_path.suffix.lower() in [".docx", ".doc"]:
                content = cls.load_docx(str(file_path))
            elif file_path.suffix.lower() in [".md", ".markdown"]:
                content = cls.load_markdown(str(file_path))
            else:
                # 尝试作为文本文件读取
                content = cls.load_txt(str(file_path))

            if not content.strip():
                return None

            doc_id = cls.generate_doc_id(content, str(file_path))
            metadata = {
                "source": str(file_path),
                "file_type": file_path.suffix.lower(),
                "file_size": file_path.stat().st_size,
            }

            return Document(
                id=doc_id, content=content, metadata=metadata, source=str(file_path)
            )

        except Exception as e:
            print(f"加载文档 {file_path} 时出错: {e}")
            return None

    @classmethod
    def chunk_document(
        cls, document: Document, chunk_size: int = 1000, overlap: int = 200
    ) -> List[Document]:
        """将文档分块"""
        content = document.content
        chunks = []

        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]

            # 确保在句子或段落边界截断
            if end < len(content):
                # 查找最近的句子结束符
                sentence_endings = [". ", "。", "! ", "！", "? ", "？", "\n\n"]
                for sep in sentence_endings:
                    last_sep = chunk.rfind(sep)
                    if last_sep > chunk_size // 2:  # 避免太小的块
                        chunk = chunk[: last_sep + len(sep)]
                        end = start + len(chunk)
                        break

            chunk_doc = Document(
                id=f"{document.id}_chunk{chunk_index}",
                content=chunk.strip(),
                metadata={**document.metadata, "chunk_index": chunk_index},
                source=document.source,
                chunk_index=chunk_index,
            )
            chunks.append(chunk_doc)

            chunk_index += 1
            start = end - overlap  # 重叠部分

            if start >= len(content):
                break

        return chunks

    @classmethod
    def load_directory(
        cls, directory_path: Union[str, Path], recursive: bool = True
    ) -> List[Document]:
        """加载目录下的所有文档"""
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"目录不存在: {directory_path}")

        documents = []
        pattern = "**/*" if recursive else "*"

        # 支持的文档格式
        supported_extensions = {".pdf", ".txt", ".md", ".markdown", ".docx", ".doc"}

        files = list(directory.glob(pattern))
        for file_path in tqdm(files, desc="加载文档"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                doc = cls.load_document(str(file_path))
                if doc:
                    chunks = cls.chunk_document(doc)
                    documents.extend(chunks)

        return documents
