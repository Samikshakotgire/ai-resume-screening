"""Tests for scoring module."""

import pytest
from src.models import Candidate, GitHubEnrichment
from src.scoring import (
    _calculate_penalties,
    _score_ai_project_depth,
    _score_cloud_fullstack,
    _score_engineering_depth,
    _score_python_backend,
    score_candidate,
)


def make_candidate(**kwargs) -> Candidate:
    defaults = {
        "filename": "test.pdf",
        "raw_text": "",
        "name": "Test Candidate",
        "skills": [],
        "projects": [],
        "project_descriptions": [],
        "work_experience": [],
        "education": [],
    }
    defaults.update(kwargs)
    return Candidate(**defaults)


class TestAIScoring:
    def test_high_ai_score_with_framework_and_rag(self):
        c = make_candidate(
            raw_text="Built a RAG pipeline with LangChain, ChromaDB vector store, and semantic search. Implemented tool calling agents with state management.",
            skills=["LangChain", "ChromaDB"],
        )
        score, strengths, concerns = _score_ai_project_depth(c)
        assert score >= 25
        assert len(strengths) >= 2

    def test_low_ai_score_no_frameworks(self):
        c = make_candidate(
            raw_text="Basic machine learning projects using scikit-learn.",
        )
        score, strengths, concerns = _score_ai_project_depth(c)
        assert score < 15

    def test_medium_ai_score_rag_only(self):
        c = make_candidate(
            raw_text="Implemented RAG with vector database and embeddings for document Q&A.",
        )
        score, strengths, concerns = _score_ai_project_depth(c)
        assert 10 <= score <= 25

    def test_ai_score_capped_at_40(self):
        c = make_candidate(
            raw_text=(
                "Built a multi-agent system with LangGraph, tool calling, state machine, "
                "RAG pipeline with ChromaDB, semantic search, evaluation metrics, "
                "async orchestration, streaming responses, memory management. "
                "Implemented production system with deployment."
            ),
        )
        score, _, _ = _score_ai_project_depth(c)
        assert score <= 40


class TestPythonScoring:
    def test_high_python_score(self):
        c = make_candidate(
            raw_text="Python, FastAPI, SQLAlchemy, PostgreSQL, async/await, pytest. Built REST APIs with FastAPI and tested with pytest.",
            skills=["Python", "FastAPI", "SQLAlchemy", "PostgreSQL"],
        )
        score, strengths, _ = _score_python_backend(c)
        assert score >= 20

    def test_low_python_score(self):
        c = make_candidate(
            raw_text="Basic Python scripting.",
        )
        score, strengths, _ = _score_python_backend(c)
        assert score < 15


class TestCloudScoring:
    def test_high_cloud_score(self):
        c = make_candidate(
            raw_text="GCP, AWS, Docker, Kubernetes, CI/CD with GitHub Actions, React frontend.",
        )
        score, strengths, _ = _score_cloud_fullstack(c)
        assert score >= 10

    def test_low_cloud_score(self):
        c = make_candidate(
            raw_text="No cloud experience.",
        )
        score, _, _ = _score_cloud_fullstack(c)
        assert score < 3


class TestEngineeringDepth:
    def test_high_depth_score(self):
        c = make_candidate(
            raw_text="Testing, caching with Redis, async, observability, monitoring, CI/CD pipeline.",
        )
        score, _, _ = _score_engineering_depth(c)
        assert score >= 4

    def test_low_depth_score(self):
        c = make_candidate(
            raw_text="Basic projects.",
        )
        score, _, _ = _score_engineering_depth(c)
        assert score < 2


class TestPenalties:
    def test_thin_wrapper_penalty(self):
        c = make_candidate(
            raw_text="Built a simple chatbot using OpenAI API calls. Basic API integration.",
        )
        penalty = _calculate_penalties(c)
        assert penalty > 0

    def test_no_penalty_for_substantial_project(self):
        c = make_candidate(
            raw_text="Implemented a RAG pipeline with retrieval, vector store, agent orchestration, and evaluation metrics.",
        )
        penalty = _calculate_penalties(c)
        assert penalty == 0

    def test_tutorial_penalty(self):
        c = make_candidate(
            raw_text="Tutorial project following along with course. Class project for assignment.",
        )
        penalty = _calculate_penalties(c)
        assert penalty > 0


class TestFullScoring:
    def test_eligible_candidate_scoring(self):
        c = make_candidate(
            raw_text=(
                "Python, FastAPI, LangChain, ChromaDB, Docker, GCP. "
                "Built a RAG pipeline with LangChain and ChromaDB vector store. "
                "Implemented tool calling agents with state management. "
                "Deployed on GCP with Docker. Used pytest for testing."
            ),
            skills=["Python", "FastAPI", "LangChain", "ChromaDB", "Docker", "GCP"],
            projects=["RAG pipeline with LangChain and ChromaDB"],
        )
        total, breakdown, strengths, concerns = score_candidate(c)
        assert total > 0
        assert total <= 100
        assert breakdown.ai_project_depth > 0
        assert breakdown.python_backend > 0

    def test_scoring_with_github_enrichment(self):
        c = make_candidate(
            raw_text="Python, LangChain, RAG pipeline.",
            skills=["Python", "LangChain"],
        )
        gh = GitHubEnrichment(
            status="success",
            recent_activity_score=3,
            repos_score=3,
            total_score=6,
            summary="Active GitHub profile",
        )
        total, breakdown, _, _ = score_candidate(c, gh)
        assert breakdown.github == 6

    def test_scoring_without_github(self):
        c = make_candidate(
            raw_text="Python, LangChain, RAG pipeline.",
        )
        total, breakdown, _, _ = score_candidate(c)
        assert breakdown.github == 0

    def test_total_score_never_exceeds_100(self):
        c = make_candidate(
            raw_text=(
                "Python, FastAPI, LangChain, LangGraph, ChromaDB, Docker, GCP, AWS, "
                "React, testing, caching, Redis, async, observability. "
                "Built multi-agent RAG system with tool calling, state, evaluation. "
                "Deployed on GCP with Docker. Used pytest, CI/CD."
            ),
            skills=["Python", "FastAPI", "LangChain", "LangGraph", "ChromaDB", "Docker", "GCP"],
        )
        total, _, _, _ = score_candidate(c)
        assert total <= 100

    def test_total_score_never_negative(self):
        c = make_candidate(
            raw_text="Basic project.",
        )
        total, _, _, _ = score_candidate(c)
        assert total >= 0
