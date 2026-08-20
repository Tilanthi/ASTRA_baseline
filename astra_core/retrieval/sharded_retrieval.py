"""
Sharded & Scattered Retrieval (Priority 3)
==========================================

Parallel search across multiple domain-scoped indexes.

Problem Solved:
- Single monolithic vector store becomes bottleneck at scale
- Large indexes have slow search latency
- Cross-domain queries get polluted by semantically similar but irrelevant content

Benefits:
- 28% latency reduction (smaller indexes, parallel execution)
- 25% precision improvement (domain isolation)
- Linear scalability vs monolithic degradation
- Natural fit for STAN's multi-domain architecture

Based on: "Building the 14 Key Pillars of Agentic AI" - Pillar 11

Example Use:
    shards = [
        DomainShard("astronomy", astro_docs),
        DomainShard("trading", trading_docs),
        DomainShard("general", general_docs),
    ]
    retriever = ShardedRetriever(shards)
    results = retriever.retrieve("telescope optics")  # Searches astronomy shard
    # Faster and more precise than monolithic search
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

from .hybrid_search import BaseRetriever, Document, HybridRetriever, VectorRetriever, TfidfRetriever


class ShardStrategy(Enum):
    """Strategy for determining which shards to query."""
    ALL = "all"                      # Query all shards
    SELECTIVE = "selective"          # Query only relevant shards
    ADAPTIVE = "adaptive"            # Adapt based on query analysis


@dataclass
class DomainShard:
    """
    A domain-scoped shard of documents.

    Each shard contains documents from a specific domain and has its own
    retriever for efficient, domain-isolated search.
    """
    name: str
    domain: str
    documents: List[Document]
    retriever: Optional[BaseRetriever] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
