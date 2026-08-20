#!/usr/bin/env python3
"""
Astronomical Paper Library - RAG Query System
==============================================

Retrieval-Augmented Generation system for querying your paper library.
Integrates with Claude/other LLMs for intelligent question-answering.

Author: STAN_IX_ASTRO
Date: January 10, 2026
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import paper library (relative import)
try:
    from .paper_library import PaperLibrary, Paper, PaperChunk
except ImportError:
    from paper_library import PaperLibrary, Paper, PaperChunk

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RetrievedContext:
    """Context retrieved from paper library."""
    chunks: List[PaperChunk]
    papers: Dict[str, Paper]
    query: str
    retrieval_method: str

    def format_for_llm(self, max_chars: int = 8000) -> str:
        """
        Format retrieved context for LLM consumption.

        Creates a structured prompt with paper citations.
        """
        context_parts = []

        context_parts.append("RELEVANT PASSAGES FROM PAPER LIBRARY:\n")
        context_parts.append("=" * 80 + "\n\n")

        # Group chunks by paper
        paper_chunks = {}
        for chunk in self.chunks:
            if chunk.paper_id not in paper_chunks:
                paper_chunks[chunk.paper_id] = []
            paper_chunks[chunk.paper_id].append(chunk)

        # Add each paper's chunks
        current_chars = 100  # Reserve for header/footer
        for paper_id, chunks in paper_chunks.items():
            if current_chars >= max_chars:
                break

            paper = self.papers.get(paper_id)
            if not paper:
                continue

            # Paper header with citation
            citation = f"{paper.authors[0] if paper.authors else 'Unknown'} et al. ({paper.year})"
            if paper.journal:
                citation += f", {paper.journal}"
            if paper.doi:
                citation += f", DOI: {paper.doi}"

            context_parts.append(f"PAPER: {paper.title}\n")
            context_parts.append(f"CITATION: {citation}\n")
            context_parts.append("-" * 80 + "\n")

            current_chars += len(citation) + 50

            # Add chunks from this paper
            for chunk in chunks:
                chunk_text = chunk.text.strip()
                if len(chunk_text) > 500:
                    chunk_text = chunk_text[:500] + "..."

                context_parts.append(f"[{chunk.metadata.get('section', 'Text')}]\n")
                context_parts.append(f"{chunk_text}\n\n")

                current_chars += len(chunk_text) + 50
                if current_chars >= max_chars:
                    break

            context_parts.append("\n")

        return "".join(context_parts[:max_chars])


@dataclass
class QueryResult:
    """Result from RAG query."""
    answer: str
    context: RetrievedContext
    sources: List[Dict[str, Any]]
    confidence: float
    query: str


# =============================================================================
# RAG Query System
# =============================================================================


class PaperRAGSystem:
    """
    RAG query interface over the astronomical paper library.

    Retrieves the most relevant passages with lexical TF-IDF search and
    packages them, with citations, either as a QueryResult or as a
    ready-to-paste LLM prompt (``format_context_for_llm``).  The answer
    field is extractive (best-matching passages with citations) - the
    generative step is delegated to the caller's LLM.
    """

    def __init__(self, library_path: Optional[str] = None):
        self.library = PaperLibrary(library_path=library_path)

    # ------------------------------------------------------------------ query
    def query(self, question: str, k: int = 5) -> QueryResult:
        """Retrieve and package context for a natural-language question."""
        results = self.library.search(question, k=k)
        chunks = [chunk for chunk, _score in results]
        papers: Dict[str, Paper] = {}
        for chunk in chunks:
            if chunk.paper_id not in papers:
                paper = self.library.get_paper(chunk.paper_id)
                if paper is not None:
                    papers[paper.paper_id] = paper

        context = RetrievedContext(
            chunks=chunks,
            papers=papers,
            query=question,
            retrieval_method="tfidf",
        )
        answer = self._extractive_answer(question, context)
        sources = [self._source_entry(chunk, score, papers)
                   for chunk, score in results]
        confidence = min(1.0, results[0][1] / 50.0) if results else 0.0

        return QueryResult(
            answer=answer,
            context=context,
            sources=sources,
            confidence=confidence,
            query=question,
        )

    def ask(self, question: str, k: int = 5) -> str:
        """Convenience wrapper returning only the answer string."""
        return self.query(question, k=k).answer

    def format_context_for_llm(self, question: str, k: int = 5,
                               max_chars: int = 8000) -> str:
        """Retrieve context formatted as an LLM prompt supplement."""
        return self.query(question, k=k).context.format_for_llm(max_chars)

    # -------------------------------------------------------------- internals
    @staticmethod
    def _citation(paper: Paper) -> str:
        first = paper.authors[0] if paper.authors else 'Unknown'
        cite = f"{first} et al. ({paper.year})"
        if paper.journal:
            cite += f", {paper.journal}"
        return cite

    def _extractive_answer(self, question: str,
                           context: RetrievedContext) -> str:
        """Best-matching passages with citations (no generation)."""
        if not context.chunks:
            return (f"No relevant passages found in the paper library "
                    f"for: '{question}'")
        parts = [f"Top passages relevant to '{question}':"]
        for chunk in context.chunks[:3]:
            paper = context.papers.get(chunk.paper_id)
            cite = self._citation(paper) if paper else chunk.paper_id
            excerpt = ' '.join(chunk.text.split())[:300]
            parts.append(f"- [{cite}] {excerpt}")
        parts.append("(Passages above are verbatim extracts; use "
                     "format_context_for_llm for the full prompt context.)")
        return "\n".join(parts)

    def _source_entry(self, chunk: PaperChunk, score: float,
                      papers: Dict[str, Paper]) -> Dict[str, Any]:
        paper = papers.get(chunk.paper_id)
        return {
            'paper_id': chunk.paper_id,
            'title': paper.title if paper else '',
            'citation': self._citation(paper) if paper else chunk.paper_id,
            'chunk_id': chunk.chunk_id,
            'score': round(float(score), 3),
        }
