"""Eligibility module — hard rule-based Python + AI/agentic filter."""

from __future__ import annotations

from src.config import AI_KEYWORDS, PYTHON_KEYWORDS
from src.models import Candidate, EligibilityResult


def _has_python_evidence(candidate: Candidate) -> tuple[bool, list[str]]:
    """
    Check for genuine Python evidence:
    - Python as a skill
    - Python mentioned in project technologies
    - Python in work/internship experience
    """
    evidence = []
    text_lower = candidate.raw_text.lower()
    skills_lower = [s.lower() for s in candidate.skills]

    # Check direct skill mentions
    for kw in PYTHON_KEYWORDS:
        if kw in skills_lower:
            evidence.append(kw)

    # Check project descriptions for Python usage
    for desc in candidate.project_descriptions + candidate.projects:
        desc_lower = desc.lower()
        for kw in PYTHON_KEYWORDS:
            if kw in desc_lower and kw not in evidence:
                evidence.append(kw)

    # Check work experience for Python usage
    for exp in candidate.work_experience:
        exp_lower = exp.lower()
        for kw in PYTHON_KEYWORDS:
            if kw in exp_lower and kw not in evidence:
                evidence.append(kw)

    # Also check raw text for strong Python signals
    python_strong_signals = ["python", "fastapi", "django", "flask", "pytorch", "numpy", "pandas"]
    for sig in python_strong_signals:
        if sig in text_lower and sig not in evidence:
            evidence.append(sig)

    return len(evidence) > 0, evidence


def _has_ai_evidence(candidate: Candidate) -> tuple[bool, list[str]]:
    """
    Check for meaningful AI/LLM/RAG/agentic evidence:
    - AI frameworks in skills (LangChain, LlamaIndex, etc.)
    - AI keywords in project descriptions (must show implementation, not just listing)
    - AI keywords in work experience
    """
    evidence = []
    text_lower = candidate.raw_text.lower()

    # Check skills for AI keywords
    skills_lower = [s.lower() for s in candidate.skills]
    ai_frameworks = [
        "langchain", "langgraph", "llamaindex", "llama-index",
        "openai", "anthropic", "hugging face", "huggingface",
        "transformers", "pytorch", "tensorflow",
        "chromadb", "pinecone", "weaviate", "qdrant", "faiss",
    ]
    for fw in ai_frameworks:
        if fw in skills_lower:
            evidence.append(fw)

    # Check project descriptions for AI implementation signals
    ai_implementation_signals = [
        "rag", "retrieval augmented", "retrieval-augmented",
        "agent", "agentic", "multi-agent", "tool calling", "function calling",
        "langchain", "langgraph", "llamaindex",
        "vector", "embedding", "semantic search",
        "llm", "large language model",
        "fine-tun", "fine tun",
        "prompt engineer",
        "chatbot", "conversational",
        "pipeline", "workflow",
        "evaluation", "eval",
    ]

    all_descriptions = candidate.project_descriptions + candidate.projects + candidate.work_experience
    for desc in all_descriptions:
        desc_lower = desc.lower()
        for sig in ai_implementation_signals:
            if sig in desc_lower:
                evidence.append(sig)

    # Check raw text for AI signals in context (projects/skills sections)
    for sig in ai_implementation_signals:
        if sig in text_lower and sig not in evidence:
            evidence.append(sig)

    # Deduplicate
    evidence = list(set(evidence))
    return len(evidence) > 0, evidence


def check_eligibility(candidate: Candidate) -> EligibilityResult:
    """
    Apply hard eligibility rules:
    1. Must have Python evidence
    2. Must have AI/agentic evidence
    """
    has_python, python_evidence = _has_python_evidence(candidate)
    has_ai, ai_evidence = _has_ai_evidence(candidate)

    matched_skills = list(set(python_evidence + ai_evidence))
    rejection_reasons = []

    if not has_python:
        rejection_reasons.append("No Python evidence found in skills, projects, or experience")
    if not has_ai:
        rejection_reasons.append("No AI/LLM/RAG/agentic project or framework evidence found")

    eligible = has_python and has_ai

    return EligibilityResult(
        eligible=eligible,
        has_python_evidence=has_python,
        has_ai_evidence=has_ai,
        rejection_reasons=rejection_reasons,
        matched_skills=matched_skills,
    )
