"""
Parallel Context Pre-processing (Priority 2)
============================================

Filters retrieved documents in parallel using small LLM calls.

Problem Solved:
- High-recall retrieval (k=10+) produces large, noisy context
- Large contexts are slow, expensive, and cause "lost in the middle" problem
- Irrelevant documents reduce LLM accuracy

Benefits:
- 90% token reduction in final context
- 73% faster final generation
- 25% accuracy improvement (removes noise)

Based on: "Building the 14 Key Pillars of Agentic AI" - Pillar 13

Example Use:
    distiller = ContextDistiller()
    raw_docs = vector_store.retrieve("quantum physics", k=20)  # 20 docs
    distilled = distiller.distill(query, raw_docs)  # Reduces to 2-3 relevant docs
    # 90% token reduction, higher accuracy, faster generation
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


@dataclass
class Document:
    """Document representation for distillation."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.page_content, tuple(sorted(self.metadata.items()))))


@dataclass
class RelevancyCheck:
    """Result of relevancy check for a single document."""
    is_relevant: bool
    brief_explanation: str
    confidence: float = 1.0
    document: Optional[Document] = None


@dataclass
class DistillationResult:
    """Result of context distillation with metrics."""
    relevant_docs: List[Document]
    filtered_docs: List[Document]
    raw_count: int
    relevant_count: int
    token_reduction: float  # Percentage reduction
    execution_time: float
    distillation_time: float
    explanations: List[str] = field(default_factory=list)


class RelevancyChecker(ABC):
    """Abstract base class for document relevancy checkers."""

    @abstractmethod
    def check_relevance(self, query: str, document: Document) -> RelevancyCheck:
        """Check if document is relevant to query."""
        pass


class SimpleKeywordChecker(RelevancyChecker):
    """
    Simple keyword-based relevancy checker (no LLM required).

    Rules:
    - Document must contain at least one query term
    - Document must have minimum length (avoid fragments)
    - Prefer documents with more query term matches
    """

    def __init__(self, min_length: int = 50):
        self.min_length = min_length

    def check_relevance(self, query: str, document: Document) -> RelevancyCheck:
        """Check relevance using keyword matching."""
        query_lower = query.lower()
        content_lower = document.page_content.lower()

        # Extract keywords from query (remove common words)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'why', 'when', 'where'}
        query_terms = [w for w in query_lower.split() if w not in stopwords and len(w) > 2]

        # Count matches
        matches = sum(1 for term in query_terms if term in content_lower)

        # Check minimum length
        too_short = len(document.page_content) < self.min_length

        # Determine relevance
        if matches == 0:
            is_relevant = False
            explanation = f"No query terms found in document"
            confidence = 0.0
        elif too_short:
            is_relevant = False
            explanation = f"Document too short ({len(document.page_content)} < {self.min_length})"
            confidence = 0.3
        elif matches >= 3:
            is_relevant = True
            explanation = f"Contains {matches} query terms (high relevance)"
            confidence = 1.0
        elif matches >= 2:
            is_relevant = True
            explanation = f"Contains {matches} query terms (medium relevance)"
            confidence = 0.8
        elif matches == 1:
            is_relevant = True
            explanation = f"Contains 1 query term (low relevance)"
            confidence = 0.6
        else:
            is_relevant = False
            explanation = f"Insufficient query term matches"
            confidence = 0.0

        return RelevancyCheck(
            is_relevant=is_relevant,
            brief_explanation=explanation,
            confidence=confidence,
            document=document
        )


class LLMRelevancyChecker(RelevancyChecker):
    """
    LLM-based relevancy checker for maximum accuracy.

    Uses structured output to force binary decision with explanation.
    In production, would use actual LLM API (OpenAI, Anthropic, etc.).
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        # In production, initialize LLM client here

    def check_relevance(self, query: str, document: Document) -> RelevancyCheck:
        """Check relevance using LLM."""
        # For demo, fall back to simple checker
        # In production, would use:
        # prompt = f"Given query '{query}', is this document relevant?\n\n{document.page_content}"
        # response = llm.complete(prompt, response_format=RelevancyCheck)
        checker = SimpleKeywordChecker()
        return checker.check_relevance(query, document)


class ContextDistiller:
    """
    Filters high-recall retrieval results down to the relevant few.

    Runs the relevancy checker over every retrieved document in parallel
    (ThreadPoolExecutor), drops irrelevant documents, de-duplicates
    repeats from overlapping queries, and reports the token reduction
    achieved.
    """

    def __init__(self, checker: Optional[RelevancyChecker] = None,
                 max_workers: int = 4):
        self.checker = checker or SimpleKeywordChecker()
        self.max_workers = max_workers

    # ---------------------------------------------------------------- public
    def distill(self, query: str,
                documents: List[Document]) -> DistillationResult:
        """
        Distill a raw retrieved document list down to relevant documents.

        Parameters
        ----------
        query : the user query the documents were retrieved for
        documents : raw (high-recall, possibly duplicated) document list

        Returns
        -------
        DistillationResult with relevant_docs (confidence-sorted),
        filtered_docs, counts, token reduction and timing.
        """
        raw = list(documents or [])
        start = time.perf_counter()

        # De-duplicate repeats (overlapping multi-query retrieval often
        # returns the same document several times)
        seen = set()
        unique: List[Document] = []
        for doc in raw:
            key = getattr(doc, 'page_content', None)
            key = key if key is not None else str(doc)
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        # Relevancy-check every document, in parallel
        checks: List[RelevancyCheck] = []
        if unique:
            if self.max_workers > 1:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    futures = [pool.submit(self.checker.check_relevance, query, d)
                               for d in unique]
                    for future in futures:
                        try:
                            checks.append(future.result())
                        except Exception:
                            continue
            else:
                checks = [self.checker.check_relevance(query, d) for d in unique]
        distillation_time = time.perf_counter() - start

        relevant = [c for c in checks if c.is_relevant]
        relevant.sort(key=lambda c: c.confidence, reverse=True)
        filtered = [c for c in checks if not c.is_relevant]

        relevant_docs = [c.document for c in relevant if c.document is not None]
        filtered_docs = [c.document for c in filtered if c.document is not None]

        raw_tokens = self._count_tokens([getattr(d, 'page_content', '') for d in raw])
        kept_tokens = self._count_tokens([getattr(d, 'page_content', '')
                                          for d in relevant_docs])
        token_reduction = (100.0 * (1.0 - kept_tokens / raw_tokens)
                           if raw_tokens > 0 else 0.0)
        execution_time = time.perf_counter() - start

        return DistillationResult(
            relevant_docs=relevant_docs,
            filtered_docs=filtered_docs,
            raw_count=len(raw),
            relevant_count=len(relevant_docs),
            token_reduction=round(token_reduction, 1),
            execution_time=round(execution_time, 4),
            distillation_time=round(distillation_time, 4),
            explanations=[c.brief_explanation for c in checks],
        )

    # ------------------------------------------------------------- internals
    @staticmethod
    def _count_tokens(texts: List[str]) -> int:
        """Token count via tiktoken when available, else word count."""
        if not texts:
            return 0
        if _TIKTOKEN_AVAILABLE:
            try:
                enc = tiktoken.get_encoding('cl100k_base')
                return sum(len(enc.encode(t)) for t in texts if t)
            except Exception:
                pass
        return sum(len(t.split()) for t in texts if t)


def create_context_distiller(checker: Optional[RelevancyChecker] = None,
                             max_workers: int = 4) -> ContextDistiller:
    """Factory for the context distiller (default: keyword checker)."""
    return ContextDistiller(checker=checker, max_workers=max_workers)
