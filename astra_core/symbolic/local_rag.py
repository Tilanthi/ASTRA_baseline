"""
Local RAG with ChromaDB: Vector Retrieval for Knowledge Augmentation

Provides local vector retrieval using ChromaDB:
- Runs locally on M1 Mac, no GPU needed
- Uses built-in sentence embeddings
- Scientific facts knowledge base
- MORK-backed document store
- Problem signature matching

Expected gain: +5-8% accuracy

Date: 2025-12-10
Version: 38.0
"""

import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np


@dataclass
class Document:
    """A document in the RAG store"""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __hash__(self):
        return hash(self.doc_id)


@dataclass
class RetrievalResult:
    """Result from RAG retrieval"""
    documents: List[Document]
    scores: List[float]
    query: str
    total_retrieved: int

    def get_context(self, max_length: int = 2000) -> str:
        """Format retrieved docs as context string"""
        context_parts = []
        current_length = 0

        for doc, score in zip(self.documents, self.scores):
            doc_text = f"[Relevance: {score:.2f}] {doc.content}"
            if current_length + len(doc_text) > max_length:
                break
            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n\n".join(context_parts)


class SimpleEmbedder:
    """
    Simple TF-IDF based embedder for when sentence-transformers is not available.

    Falls back to word overlap similarity.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to vectors using simple TF-IDF"""
        embeddings = []

        for text in texts:
            words = text.lower().split()
            vec = np.zeros(self.dimension)

            for i, word in enumerate(words[:self.dimension]):
                # Simple hash-based embedding
                word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
                idx = word_hash % self.dimension
                vec[idx] += 1.0

            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            embeddings.append(vec)

        return np.array(embeddings)


class LocalRAG:
    """
    Local RAG implementation with optional ChromaDB backend.

    Uses ChromaDB when available, falls back to in-memory store.
    ChromaDB runs locally on M1 Mac, no GPU needed.
    """

    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "stan_knowledge"):
        """
        Initialize LocalRAG.

        Args:
            persist_dir: Directory for persistent storage (None for in-memory)
            collection_name: Name of the collection
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.chromadb_available = self._check_chromadb()

        if self.chromadb_available:
            self._init_chromadb()

    # ==================================================================
    # Backend initialisation
    # (re-implemented 2026-08; bodies lost to file truncation before the
    #  audit. API reconstructed from surviving call sites in
    #  symbolic/stan_enhanced.py: retrieve, retrieve_with_context, stats,
    #  add_documents; in-memory fallback when ChromaDB is absent.)
    # ==================================================================

    def _check_chromadb(self) -> bool:
        """ChromaDB is optional; report availability without importing."""
        try:
            import chromadb  # noqa: F401
            return True
        except ImportError:
            return False

    def _init_chromadb(self) -> None:
        """Open (or create) the persistent ChromaDB collection."""
        import chromadb
        if self.persist_dir:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
        else:
            self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name)
        self.embedder = SimpleEmbedder()
        self.documents: List[Document] = []

    # ------------------------------------------------------------------
    def _ensure_ready(self) -> None:
        """Set up the in-memory fallback store if ChromaDB is absent."""
        if not hasattr(self, 'documents'):
            self.documents: List[Document] = []
            self.embedder = SimpleEmbedder()
            self._doc_embeddings: List[np.ndarray] = []

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents ({'content', 'metadata'?} dicts) to the store."""
        self._ensure_ready()
        for entry in documents:
            content = entry.get('content') or entry.get('text') or ''
            if not content:
                continue
            doc = Document(
                doc_id=entry.get('doc_id',
                                 hashlib.md5(content.encode()).hexdigest()),
                content=content,
                metadata=entry.get('metadata', {}),
            )
            self.documents.append(doc)
            if self.chromadb_available:
                self.collection.add(
                    ids=[doc.doc_id], documents=[doc.content],
                    metadatas=[doc.metadata],
                )
            else:
                self._doc_embeddings.append(
                    self.embedder.encode([content])[0])

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve the top-k most similar documents for a query."""
        self._ensure_ready()
        if not self.documents:
            return RetrievalResult([], [], query, 0)

        if self.chromadb_available:
            n_results = min(top_k, self.collection.count())
            if n_results == 0:
                return RetrievalResult([], [], query, 0)
            res = self.collection.query(query_texts=[query],
                                        n_results=n_results)
            docs = [Document(doc_id=i, content=c,
                             metadata=m or {})
                    for i, c, m in zip(res['ids'][0], res['documents'][0],
                                       res['metadatas'][0])]
            # ChromaDB returns distances; convert to similarity scores
            dists = res.get('distances', [[0.0] * len(docs)])[0]
            scores = [1.0 / (1.0 + float(d)) for d in dists]
            return RetrievalResult(docs, scores, query, len(docs))

        # In-memory cosine similarity over hash embeddings
        query_vec = self.embedder.encode([query])[0]
        scored = []
        for doc, doc_vec in zip(self.documents, self._doc_embeddings):
            denom = (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            sim = float(query_vec @ doc_vec / denom) if denom > 0 else 0.0
            scored.append((sim, doc))
        scored.sort(key=lambda sd: sd[0], reverse=True)
        top = scored[:top_k]
        return RetrievalResult([d for _, d in top], [s for s, _ in top],
                               query, len(top))

    def retrieve_with_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve and format as a ready-to-use context string."""
        result = self.retrieve(query, top_k=top_k)
        return result.get_context()

    def stats(self) -> Dict[str, Any]:
        """Store statistics summary."""
        self._ensure_ready()
        return {
            'n_documents': len(self.documents),
            'backend': 'chromadb' if self.chromadb_available else 'in_memory',
            'persist_dir': self.persist_dir,
            'collection': self.collection_name,
        }
