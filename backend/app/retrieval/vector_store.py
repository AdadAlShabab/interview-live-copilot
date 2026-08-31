"""Optional local FAISS index for career evidence."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvidenceChunk:
    """A searchable piece of candidate evidence."""

    text: str
    source: str
    metadata: dict[str, Any]


def candidate_evidence(candidate: Any) -> list[EvidenceChunk]:
    """Flatten a candidate ORM object into focused, searchable evidence chunks."""
    chunks: list[EvidenceChunk] = []
    for skill in candidate.skills or []:
        chunks.append(EvidenceChunk(skill, "skill", {"skill": skill}))
    if candidate.summary:
        chunks.append(EvidenceChunk(candidate.summary, "summary", {}))
    for experience in candidate.experiences:
        details = [
            f"{experience.role} at {experience.company}",
            *(experience.responsibilities or []),
            *(experience.technologies or []),
            *(experience.achievements or []),
            *(experience.projects or []),
        ]
        chunks.append(EvidenceChunk(". ".join(details), "experience", {
            "company": experience.company,
            "role": experience.role,
        }))
    for story in candidate.stories:
        text = ". ".join([
            story.title,
            story.situation,
            story.task,
            story.action,
            story.result,
        ])
        chunks.append(EvidenceChunk(text, "behavioral_story", {"theme": story.theme}))
    return chunks


class VectorStore:
    """Persist evidence and use FAISS when its optional dependencies exist."""

    def __init__(self, index_dir: Path, embedder: Any | None = None):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "evidence.index"
        self.metadata_path = self.index_dir / "evidence.json"
        self.embedder = embedder
        self.chunks: list[EvidenceChunk] = []
        self._index = None
        self._faiss = None
        self._load_optional_faiss()
        self.load()

    @property
    def semantic_available(self) -> bool:
        return self._index is not None and self.embedder is not None

    def add(self, chunks: list[EvidenceChunk]) -> None:
        """Replace the current evidence set and persist it locally."""
        self.chunks = list(chunks)
        self._index = None
        if self.embedder is not None and self._faiss is not None and self.chunks:
            vectors = self._embed([chunk.text for chunk in self.chunks])
            self._index = self._faiss.IndexFlatIP(vectors.shape[1])
            self._index.add(vectors)
        self.save()

    def search(self, query: str, limit: int = 3) -> list[tuple[EvidenceChunk, float]]:
        """Return semantic matches, or an empty list when FAISS is unavailable."""
        if not self.semantic_available or not query.strip() or limit < 1:
            return []
        vectors = self._embed([query])
        scores, indices = self._index.search(vectors, min(limit, len(self.chunks)))
        return [
            (self.chunks[index], float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            json.dumps([asdict(chunk) for chunk in self.chunks], ensure_ascii=True),
            encoding="utf-8",
        )
        if self._index is not None:
            self._faiss.write_index(self._index, str(self.index_path))

    def load(self) -> None:
        if self.metadata_path.exists():
            values = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.chunks = [EvidenceChunk(**value) for value in values]
        if (
            self._faiss is not None
            and self.embedder is not None
            and self.index_path.exists()
            and self.chunks
        ):
            self._index = self._faiss.read_index(str(self.index_path))

    def _load_optional_faiss(self) -> None:
        try:
            import faiss
        except ImportError:
            return
        self._faiss = faiss

    def _embed(self, texts: list[str]):
        import numpy as np

        vectors = np.asarray(self.embedder.encode(texts), dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)