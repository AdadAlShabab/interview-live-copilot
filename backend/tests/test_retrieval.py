"""Tests for local evidence retrieval."""

from app.retrieval.retriever import EvidenceRetriever
from app.retrieval.vector_store import EvidenceChunk, VectorStore, candidate_evidence


def test_keyword_fallback_ranks_relevant_evidence(tmp_path):
    store = VectorStore(tmp_path / "faiss")
    store.add([
        EvidenceChunk("Built Python APIs with FastAPI", "experience", {}),
        EvidenceChunk("Led a cross-functional hiring project", "story", {}),
    ])

    results = EvidenceRetriever(store).retrieve("How did you build Python APIs?", limit=1)

    assert len(results) == 1
    assert results[0].source == "experience"


def test_store_persists_chunks_without_faiss(tmp_path):
    path = tmp_path / "faiss"
    VectorStore(path).add([EvidenceChunk("PostgreSQL optimization", "skill", {"name": "SQL"})])

    restored = VectorStore(path)

    assert restored.chunks[0].text == "PostgreSQL optimization"
    assert restored.chunks[0].metadata["name"] == "SQL"


def test_candidate_evidence_flattens_resume_sections():
    candidate = type("Candidate", (), {
        "skills": ["Python"],
        "summary": "Backend engineer",
        "experiences": [type("Experience", (), {
            "company": "Acme",
            "role": "Engineer",
            "responsibilities": ["Built APIs"],
            "technologies": ["FastAPI"],
            "achievements": [],
            "projects": [],
        })()],
        "stories": [],
    })()

    chunks = candidate_evidence(candidate)

    assert {chunk.source for chunk in chunks} == {"skill", "summary", "experience"}
    assert any("Built APIs" in chunk.text for chunk in chunks)