"""
检索器模块
"""

import chromadb
from chromadb.config import Settings
from typing import Any, List, Optional
import dspy


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
