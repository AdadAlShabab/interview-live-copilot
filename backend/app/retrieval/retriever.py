"""Hybrid evidence retrieval with a dependency-free keyword fallback."""

import re

from app.retrieval.vector_store import EvidenceChunk, VectorStore


class EvidenceRetriever:
    """Retrieve the most relevant candidate evidence for a question."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, limit: int = 3) -> list[EvidenceChunk]:
        if limit < 1 or not query.strip():
            return []

        semantic = self.vector_store.search(query, limit=limit)
        keyword = self._keyword_matches(query, limit)
        ranked: dict[str, tuple[EvidenceChunk, float]] = {}
        for chunk, score in semantic:
            ranked[chunk.text] = (chunk, score + 0.25)
        for chunk, score in keyword:
            existing = ranked.get(chunk.text)
            ranked[chunk.text] = (chunk, max(score, existing[1] if existing else 0))
        return [chunk for chunk, _ in sorted(ranked.values(), key=lambda item: item[1], reverse=True)[:limit]]

    def _keyword_matches(self, query: str, limit: int) -> list[tuple[EvidenceChunk, float]]:
        query_terms = self._terms(query)
        if not query_terms:
            return []
        matches = []
        for chunk in self.vector_store.chunks:
            chunk_terms = self._terms(chunk.text)
            overlap = len(query_terms & chunk_terms)
            if overlap:
                matches.append((chunk, overlap / len(query_terms)))
        return sorted(matches, key=lambda item: item[1], reverse=True)[:limit]

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {term for term in re.findall(r"[a-z0-9+#.-]+", text.lower()) if len(term) > 2}