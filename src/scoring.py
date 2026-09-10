"""Scoring module — 100-point weighted scoring + penalties for eligible candidates."""

from __future__ import annotations

import re
from typing import Optional

from src.config import (
    AI_KEYWORDS,
    CLOUD_KEYWORDS,
    ENGINEERING_DEPTH_KEYWORDS,
    PENALTY_MAX,
    PENALTY_MIN,
    THIN_WRAPPER_PENALTY,
    TUTORIAL_PENALTY,
    WEIGHT_AI_PROJECT_DEPTH,
    WEIGHT_CLOUD_FULLSTACK,
    WEIGHT_ENGINEERING_DEPTH,
    WEIGHT_GITHUB,
    WEIGHT_PYTHON_BACKEND,
)
from src.models import Candidate, GitHubEnrichment, ScoreBreakdown


def _score_ai_project_depth(candidate: Candidate) -> tuple[float, list[str], list[str]]:
    """
    Score 0-40 for AI/Agentic/RAG project depth.
    Rewards real systems: agents, RAG, tools, retrieval, state, orchestration, eval.
    """
    score = 0.0
    strengths = []
    concerns = []
    text_lower = candidate.raw_text.lower()

    # Strong AI framework usage signals
    strong_frameworks = [
        "langchain", "langgraph", "llamaindex", "llama-index",
        "google adk", "google agent development kit",
    ]
    framework_found = False
    for fw in strong_frameworks:
        if fw in text_lower:
            framework_found = True
            break

    # RAG pipeline signals
    rag_signals = [
        "rag", "retrieval augmented", "retrieval-augmented",
        "vector store", "vector database", "embedding",
        "semantic search", "chunking", "retrieval pipeline",
    ]
    rag_count = sum(1 for s in rag_signals if s in text_lower)

    # Agentic signals
    agent_signals = [
        "agent", "agentic", "multi-agent", "tool calling", "function calling",
        "state machine", "orchestration", "workflow",
        "memory", "conversation history",
    ]
    agent_count = sum(1 for s in agent_signals if s in text_lower)

    # Evaluation/testing signals
    eval_signals = ["evaluation", "eval", "benchmark", "metrics", "testing", "accuracy"]
    eval_count = sum(1 for s in eval_signals if s in text_lower)

    # Real implementation detail (not just keyword listing)
    implementation_detail = False
    detail_signals = [
        "implemented", "built", "developed", "designed", "deployed",
        "production", "pipeline", "system", "architecture",
        "async", "concurrent", "streaming",
    ]
    for sig in detail_signals:
        if sig in text_lower:
            implementation_detail = True
            break

    # Score components
    if framework_found:
        score += 10
        strengths.append("Uses AI framework(s) beyond basic API calls")

    if rag_count >= 2:
        score += min(15, 5 + rag_count * 3)
        strengths.append(f"RAG implementation with {rag_count} relevant signals")
    elif rag_count == 1:
        score += 5

    if agent_count >= 2:
        score += min(12, 4 + agent_count * 3)
        strengths.append(f"Agentic workflow with {agent_count} relevant signals")
    elif agent_count == 1:
        score += 4

    if eval_count >= 2:
        score += 5
        strengths.append("Evaluation/metrics for AI pipeline")

    if implementation_detail:
        score += 5
        strengths.append("Shows implementation detail beyond keyword listing")

    # Cap at 40
    score = min(score, 40)

    if score < 10:
        concerns.append("Limited AI project depth evidence")

    return score, strengths, concerns


def _score_python_backend(candidate: Candidate) -> tuple[float, list[str], list[str]]:
    """
    Score 0-30 for Python & Backend engineering.
    Rewards project/internship evidence over keyword-only skill lists.
    """
    score = 0.0
    strengths = []
    concerns = []
    text_lower = candidate.raw_text.lower()

    # Python-specific frameworks
    backend_frameworks = ["fastapi", "django", "flask", "uvicorn", "sqlalchemy", "pydantic"]
    framework_count = sum(1 for fw in backend_frameworks if fw in text_lower)

    # Backend services
    backend_services = ["postgresql", "postgres", "mysql", "redis", "celery", "rabbitmq", "kafka"]
    service_count = sum(1 for s in backend_services if s in text_lower)

    # Async/concurrency
    async_signals = ["async", "asyncio", "async/await", "concurrent", "parallel"]
    async_count = sum(1 for s in async_signals if s in text_lower)

    # Testing
    test_signals = ["pytest", "unittest", "testing", "test driven", "tdd", "coverage"]
    test_count = sum(1 for s in test_signals if s in text_lower)

    # Project/internship evidence of Python usage
    has_project_evidence = False
    for desc in candidate.project_descriptions + candidate.projects:
        desc_lower = desc.lower()
        if any(fw in desc_lower for fw in ["python", "fastapi", "django", "flask", "pytorch"]):
            has_project_evidence = True
            break

    has_work_evidence = False
    for exp in candidate.work_experience:
        exp_lower = exp.lower()
        if any(fw in exp_lower for fw in ["python", "fastapi", "django", "flask", "pytorch"]):
            has_work_evidence = True
            break

    # Score components
    if framework_count >= 2:
        score += 10
        strengths.append(f"Multiple Python backend frameworks ({framework_count})")
    elif framework_count == 1:
        score += 6
        strengths.append("Python backend framework usage")

    if service_count >= 2:
        score += 8
        strengths.append(f"Backend service experience ({service_count} services)")
    elif service_count == 1:
        score += 4

    if async_count >= 1:
        score += 4
        strengths.append("Async/concurrency experience")

    if test_count >= 1:
        score += 4
        strengths.append("Testing practice evidence")

    if has_project_evidence:
        score += 4
        strengths.append("Python used in project implementation")

    if has_work_evidence:
        score += 3
        strengths.append("Python used in work experience")

    score = min(score, 30)

    if score < 8:
        concerns.append("Limited Python backend evidence")

    return score, strengths, concerns


def _score_cloud_fullstack(candidate: Candidate) -> tuple[float, list[str], list[str]]:
    """
    Score 0-15 for Cloud/Deployment/Full-stack.
    React/Next.js as supporting signal within an end-to-end system.
    """
    score = 0.0
    strengths = []
    concerns = []
    text_lower = candidate.raw_text.lower()

    # Cloud platforms
    cloud_platforms = ["gcp", "google cloud", "aws", "amazon web services", "azure"]
    cloud_count = sum(1 for p in cloud_platforms if p in text_lower)

    # DevOps/deployment
    devops = ["docker", "kubernetes", "k8s", "terraform", "ci/cd", "github actions", "jenkins"]
    devops_count = sum(1 for d in devops if d in text_lower)

    # Frontend (supporting signal)
    frontend = ["react", "next.js", "nextjs", "vue", "angular", "typescript"]
    frontend_count = sum(1 for f in frontend if f in text_lower)

    # Serverless
    serverless = ["cloud functions", "lambda", "cloud run", "serverless"]
    serverless_count = sum(1 for s in serverless if s in text_lower)

    # Score components
    if cloud_count >= 2:
        score += 5
        strengths.append(f"Multi-cloud experience ({cloud_count} platforms)")
    elif cloud_count == 1:
        score += 3
        strengths.append("Cloud platform experience")

    if devops_count >= 2:
        score += 5
        strengths.append(f"DevOps/deployment tools ({devops_count})")
    elif devops_count == 1:
        score += 3

    if frontend_count >= 1:
        score += 2
        strengths.append("Frontend/full-stack capability")

    if serverless_count >= 1:
        score += 2
        strengths.append("Serverless deployment experience")

    score = min(score, 15)

    if score < 3:
        concerns.append("Limited cloud/deployment evidence")

    return score, strengths, concerns


def _score_engineering_depth(candidate: Candidate) -> tuple[float, list[str], list[str]]:
    """
    Score 0-5 for Engineering depth signals.
    Testing, architecture, caching, queues, observability, concurrency.
    """
    score = 0.0
    strengths = []
    text_lower = candidate.raw_text.lower()

    depth_signals = 0
    for kw in ENGINEERING_DEPTH_KEYWORDS:
        if kw.lower() in text_lower:
            depth_signals += 1

    if depth_signals >= 4:
        score = 5
        strengths.append("Strong engineering depth signals")
    elif depth_signals >= 2:
        score = 3
        strengths.append("Some engineering depth signals")
    elif depth_signals >= 1:
        score = 1

    return score, strengths, []


def _calculate_penalties(candidate: Candidate) -> float:
    """Calculate penalties for thin wrappers or tutorial-style projects."""
    penalty = 0.0
    text_lower = candidate.raw_text.lower()

    # Check for thin LLM/API wrappers
    thin_wrapper_signals = [
        "api call", "api calls", "openai api", "anthropic api",
        "simple chatbot", "basic chatbot",
    ]
    thin_count = sum(1 for s in thin_wrapper_signals if s in text_lower)

    # Check if there's substance beyond API calls
    substance_signals = [
        "retrieval", "rag", "agent", "tool", "state", "workflow",
        "pipeline", "orchestration", "evaluation", "memory",
        "database", "vector", "embedding",
    ]
    substance_count = sum(1 for s in substance_signals if s in text_lower)

    if thin_count >= 2 and substance_count < 2:
        penalty += THIN_WRAPPER_PENALTY

    # Check for tutorial-style projects
    tutorial_signals = [
        "tutorial", "followed along", "following along",
        "step by step", "step-by-step",
        "course project", "class project", "homework",
    ]
    tutorial_count = sum(1 for s in tutorial_signals if s in text_lower)

    # Check for ownership evidence
    ownership_signals = [
        "built", "developed", "designed", "implemented", "created",
        "deployed", "maintained", "architected", "led",
    ]
    ownership_count = sum(1 for s in ownership_signals if s in text_lower)

    if tutorial_count >= 2 and ownership_count < 2:
        penalty += TUTORIAL_PENALTY

    return min(penalty, PENALTY_MAX)


def score_candidate(
    candidate: Candidate,
    github_enrichment: Optional[GitHubEnrichment] = None,
) -> tuple[float, ScoreBreakdown, list[str], list[str]]:
    """
    Score an eligible candidate out of 100.
    Returns (total_score, breakdown, strengths, concerns).
    """
    # Score each category
    ai_score, ai_strengths, ai_concerns = _score_ai_project_depth(candidate)
    py_score, py_strengths, py_concerns = _score_python_backend(candidate)
    cloud_score, cloud_strengths, cloud_concerns = _score_cloud_fullstack(candidate)
    eng_score, eng_strengths, eng_concerns = _score_engineering_depth(candidate)

    # GitHub score
    gh_score = 0.0
    gh_summary = ""
    if github_enrichment and github_enrichment.status == "success":
        gh_score = github_enrichment.total_score
        gh_summary = github_enrichment.summary
    elif github_enrichment and github_enrichment.status == "no_profile":
        gh_summary = "No public GitHub profile found"

    # Build breakdown
    breakdown = ScoreBreakdown(
        ai_project_depth=ai_score,
        python_backend=py_score,
        cloud_fullstack=cloud_score,
        github=gh_score,
        engineering_depth=eng_score,
    )

    # Apply penalties
    penalty = _calculate_penalties(candidate)

    # Total
    raw_total = (
        breakdown.ai_project_depth
        + breakdown.python_backend
        + breakdown.cloud_fullstack
        + breakdown.github
        + breakdown.engineering_depth
    )
    total = max(0, raw_total - penalty)

    # Collect all strengths and concerns
    all_strengths = ai_strengths + py_strengths + cloud_strengths + eng_strengths
    all_concerns = ai_concerns + py_concerns + cloud_concerns + eng_concerns

    if penalty > 0:
        all_concerns.append(f"Penalty of {penalty:.0f} applied (thin wrapper/tutorial)")

    return total, breakdown, all_strengths, all_concerns
