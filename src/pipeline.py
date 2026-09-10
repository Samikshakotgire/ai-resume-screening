"""Pipeline module — orchestrates the full resume screening run."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from src.config import (
    LLM_ENABLED,
    MAX_CONCURRENT_GITHUB,
    MAX_CONCURRENT_LLM,
)
from src.extraction import extract_candidate
from src.eligibility import check_eligibility
from src.github_enrichment import enrich_github
from src.ingestion import load_resumes
from src.llm_adapter import extract_candidate_with_llm, score_with_llm
from src.models import (
    BatchSummary,
    CandidateResult,
    GitHubEnrichment,
    PipelineOutput,
)
from src.scoring import score_candidate

logger = logging.getLogger(__name__)


def _process_single_resume(
    resume_data: dict,
    use_llm: bool = False,
) -> tuple[CandidateResult, bool]:
    """
    Process a single resume through the full pipeline.
    Returns (result, is_eligible).
    """
    filename = resume_data["filename"]
    raw_text = resume_data["raw_text"]

    # Step 1: Extract candidate info
    if use_llm and LLM_ENABLED:
        llm_data = extract_candidate_with_llm(filename, raw_text)
        if llm_data:
            candidate = extract_candidate(filename, raw_text)
            # Override with LLM extraction if available
            if llm_data.get("name"):
                candidate.name = llm_data["name"]
            if llm_data.get("email"):
                candidate.email = llm_data["email"]
            if llm_data.get("github_url"):
                candidate.github_url = llm_data["github_url"]
            if llm_data.get("skills"):
                candidate.skills = llm_data["skills"]
            if llm_data.get("projects"):
                candidate.projects = llm_data["projects"]
            if llm_data.get("project_descriptions"):
                candidate.project_descriptions = llm_data["project_descriptions"]
        else:
            candidate = extract_candidate(filename, raw_text)
    else:
        candidate = extract_candidate(filename, raw_text)

    # Step 2: Check eligibility
    eligibility = check_eligibility(candidate)

    # Build result
    result = CandidateResult(
        candidate_name=candidate.name or filename.replace(".pdf", "").replace("_", " ").title(),
        eligible=eligibility.eligible,
        matched_skills=eligibility.matched_skills,
        filename=filename,
        rejection_reasons=eligibility.rejection_reasons,
    )

    if not eligibility.eligible:
        return result, False

    # Step 3: GitHub enrichment (only for eligible candidates)
    github_enrichment = enrich_github(candidate)

    # Step 4: Score
    if use_llm and LLM_ENABLED:
        llm_result = score_with_llm(
            candidate,
            eligibility_info=f"Python: {eligibility.has_python_evidence}, AI: {eligibility.has_ai_evidence}",
        )
        if llm_result:
            total, breakdown, strengths, concerns = llm_result
            result.total_score = total
            result.score_breakdown = breakdown
            result.strengths = strengths
            result.concerns = concerns
        else:
            # Fallback to rule-based scoring
            total, breakdown, strengths, concerns = score_candidate(candidate, github_enrichment)
            result.total_score = total
            result.score_breakdown = breakdown
            result.strengths = strengths
            result.concerns = concerns
    else:
        total, breakdown, strengths, concerns = score_candidate(candidate, github_enrichment)
        result.total_score = total
        result.score_breakdown = breakdown
        result.strengths = strengths
        result.concerns = concerns

    result.github_enrichment = github_enrichment
    result.github_summary = github_enrichment.summary

    # Project summary
    if candidate.projects:
        result.project_summary = "; ".join(candidate.projects[:3])
    elif candidate.project_descriptions:
        result.project_summary = "; ".join(candidate.project_descriptions[:3])

    return result, True


def run_pipeline(
    input_dir: Path,
    output_path: Optional[Path] = None,
    use_llm: bool = False,
) -> PipelineOutput:
    """
    Run the full resume screening pipeline.
    """
    logger.info(f"Starting pipeline with input: {input_dir}")

    # Step 1: Load resumes
    resumes = load_resumes(input_dir)
    logger.info(f"Loaded {len(resumes)} resumes")

    summary = BatchSummary(
        total_resumes=len(resumes),
        successfully_parsed=sum(1 for r in resumes if r["raw_text"]),
        failed_unreadable=sum(1 for r in resumes if not r["raw_text"]),
    )

    # Step 2: Process each resume with bounded concurrency
    results: list[CandidateResult] = []
    rejected: list[CandidateResult] = []

    max_workers = min(MAX_CONCURRENT_GITHUB, len(resumes)) if resumes else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_resume, resume, use_llm): resume
            for resume in resumes
        }

        for future in as_completed(futures):
            resume = futures[future]
            try:
                result, is_eligible = future.result()
                if is_eligible:
                    results.append(result)
                else:
                    rejected.append(result)
            except Exception as e:
                logger.error(f"Failed to process {resume['filename']}: {e}")
                rejected.append(
                    CandidateResult(
                        candidate_name=resume["filename"].replace(".pdf", ""),
                        eligible=False,
                        filename=resume["filename"],
                        rejection_reasons=[f"Processing error: {str(e)}"],
                    )
                )

    # Step 3: Rank eligible candidates
    results.sort(key=lambda x: x.total_score, reverse=True)
    for i, result in enumerate(results, 1):
        result.rank = i

    summary.eligible = len(results)
    summary.rejected = len(rejected)

    output = PipelineOutput(
        results=results,
        rejected=rejected,
        summary=summary,
    )

    logger.info(
        f"Pipeline complete: {summary.eligible} eligible, "
        f"{summary.rejected} rejected, "
        f"{summary.failed_unreadable} failed/unreadable"
    )

    return output
