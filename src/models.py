"""Pydantic schemas for candidate, eligibility, and scoring."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """Represents a single parsed candidate from a resume."""

    filename: str
    raw_text: str = ""
    name: str = ""
    email: str = ""
    phone: str = ""
    github_url: str = ""
    github_username: str = ""
    skills: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    project_descriptions: list[str] = Field(default_factory=list)
    work_experience: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)


class EligibilityResult(BaseModel):
    """Outcome of eligibility check for a candidate."""

    eligible: bool
    has_python_evidence: bool = False
    has_ai_evidence: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    """Detailed scoring breakdown."""

    ai_project_depth: float = 0
    python_backend: float = 0
    cloud_fullstack: float = 0
    github: float = 0
    engineering_depth: float = 0


class GitHubEnrichment(BaseModel):
    """GitHub activity enrichment data."""

    status: str = "not_attempted"  # not_attempted | success | failed | no_profile
    recent_activity_score: float = 0
    repos_score: float = 0
    total_score: float = 0
    summary: str = ""
    recent_repos_count: int = 0
    maintained_repos_count: int = 0
    last_active_date: str = ""


class CandidateResult(BaseModel):
    """Final output for a single candidate."""

    rank: int = 0
    candidate_name: str
    eligible: bool
    total_score: float = 0
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    matched_skills: list[str] = Field(default_factory=list)
    project_summary: str = ""
    github_summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    filename: str = ""
    github_enrichment: GitHubEnrichment = Field(default_factory=GitHubEnrichment)


class BatchSummary(BaseModel):
    """Summary statistics for the entire batch."""

    total_resumes: int = 0
    successfully_parsed: int = 0
    eligible: int = 0
    rejected: int = 0
    failed_unreadable: int = 0


class PipelineOutput(BaseModel):
    """Complete output of the pipeline."""

    results: list[CandidateResult] = Field(default_factory=list)
    rejected: list[CandidateResult] = Field(default_factory=list)
    summary: BatchSummary = Field(default_factory=BatchSummary)
