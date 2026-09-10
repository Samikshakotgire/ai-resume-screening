"""Tests for eligibility module."""

import pytest
from src.models import Candidate
from src.eligibility import check_eligibility


def make_candidate(**kwargs) -> Candidate:
    defaults = {
        "filename": "test.pdf",
        "raw_text": "",
        "name": "Test Candidate",
        "email": "test@example.com",
        "skills": [],
        "projects": [],
        "project_descriptions": [],
        "work_experience": [],
        "education": [],
    }
    defaults.update(kwargs)
    return Candidate(**defaults)


class TestEligibility:
    def test_eligible_candidate_with_python_and_ai(self):
        c = make_candidate(
            raw_text="Skills: Python, LangChain, FastAPI\nProject: Built a RAG pipeline with LangChain and vector embeddings.",
            skills=["Python", "LangChain", "FastAPI"],
            projects=["Built a RAG pipeline with LangChain and vector embeddings."],
        )
        result = check_eligibility(c)
        assert result.eligible is True
        assert result.has_python_evidence is True
        assert result.has_ai_evidence is True
        assert len(result.rejection_reasons) == 0

    def test_rejected_no_python(self):
        c = make_candidate(
            raw_text="Skills: Java, React, Node.js\nProject: Built a web app with React and Express.",
            skills=["Java", "React", "Node.js"],
            projects=["Built a web app with React and Express."],
        )
        result = check_eligibility(c)
        assert result.eligible is False
        assert result.has_python_evidence is False
        assert any("Python" in r for r in result.rejection_reasons)

    def test_rejected_no_ai(self):
        c = make_candidate(
            raw_text="Skills: Python, Django, PostgreSQL\nProject: Built a REST API with Django.",
            skills=["Python", "Django", "PostgreSQL"],
            projects=["Built a REST API with Django."],
        )
        result = check_eligibility(c)
        assert result.eligible is False
        assert result.has_ai_evidence is False
        assert any("AI" in r or "agentic" in r.lower() for r in result.rejection_reasons)

    def test_rejected_empty_resume(self):
        c = make_candidate(raw_text="")
        result = check_eligibility(c)
        assert result.eligible is False
        assert len(result.rejection_reasons) == 2

    def test_eligible_with_ai_in_projects_not_skills(self):
        c = make_candidate(
            raw_text="Skills: Python, Flask\nProject: Developed a chatbot using OpenAI API with tool calling and memory.",
            skills=["Python", "Flask"],
            projects=["Developed a chatbot using OpenAI API with tool calling and memory."],
        )
        result = check_eligibility(c)
        assert result.eligible is True
        assert result.has_ai_evidence is True

    def test_eligible_with_rag_in_raw_text(self):
        c = make_candidate(
            raw_text="Skills: Python, LangChain\nBuilt a RAG pipeline with ChromaDB and semantic search.",
            skills=["Python", "LangChain"],
        )
        result = check_eligibility(c)
        assert result.eligible is True

    def test_eligible_python_from_keyword_not_skill_list(self):
        c = make_candidate(
            raw_text="Experience: Developed microservices using Python and FastAPI with async endpoints.",
            skills=[],
            work_experience=["Developed microservices using Python and FastAPI with async endpoints."],
        )
        result = check_eligibility(c)
        assert result.has_python_evidence is True

    def test_rejected_only_non_ai_frameworks(self):
        c = make_candidate(
            raw_text="Skills: Python, Django, React\nProject: Full-stack web application.",
            skills=["Python", "Django", "React"],
            projects=["Full-stack web application."],
        )
        result = check_eligibility(c)
        assert result.eligible is False
        assert result.has_python_evidence is True
        assert result.has_ai_evidence is False

    def test_eligible_with_google_adk(self):
        c = make_candidate(
            raw_text="Skills: Python, Google ADK\nProject: Multi-agent workflow using Google Agent Development Kit.",
            skills=["Python", "Google ADK"],
            projects=["Multi-agent workflow using Google Agent Development Kit."],
        )
        result = check_eligibility(c)
        assert result.eligible is True
        matched_lower = [s.lower() for s in result.matched_skills]
        assert any("google" in s or "adk" in s or "agent" in s for s in matched_lower)

    def test_matched_skills_populated(self):
        c = make_candidate(
            raw_text="Skills: Python, LangChain, Docker\nProject: RAG system with LangChain.",
            skills=["Python", "LangChain", "Docker"],
            projects=["RAG system with LangChain."],
        )
        result = check_eligibility(c)
        assert len(result.matched_skills) > 0
