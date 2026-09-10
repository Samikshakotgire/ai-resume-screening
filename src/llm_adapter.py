"""LLM adapter module — thin, swappable wrapper around LLM providers."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Any, Optional

from src.config import LLM_API_KEY, LLM_ENABLED, LLM_MODEL, LLM_PROVIDER, LLM_TIMEOUT_SECONDS
from src.models import Candidate, ScoreBreakdown

logger = logging.getLogger(__name__)

# --- Rate limiter (per-minute sliding window) ---
_rate_limit_lock = threading.Lock()
_request_timestamps: deque[float] = deque()
RATE_LIMIT_MAX = 15  # max requests per minute
RATE_LIMIT_WINDOW = 60  # window in seconds


def _wait_for_rate_limit():
    """Block until a request slot is available within the rate limit."""
    while True:
        with _rate_limit_lock:
            now = time.time()
            # Purge timestamps outside the window
            while _request_timestamps and _request_timestamps[0] < now - RATE_LIMIT_WINDOW:
                _request_timestamps.popleft()

            if len(_request_timestamps) < RATE_LIMIT_MAX:
                _request_timestamps.append(now)
                return  # slot available

            # Calculate wait time until the oldest request expires
            wait_until = _request_timestamps[0] + RATE_LIMIT_WINDOW
            wait_seconds = wait_until - now

        logger.info(f"Rate limit: waiting {wait_seconds:.1f}s for next slot")
        time.sleep(wait_seconds)


def _call_openai(prompt: str, system_prompt: str = "") -> Optional[str]:
    """Call OpenAI API and return the response text."""
    try:
        import openai
        client = openai.OpenAI(api_key=LLM_API_KEY, timeout=LLM_TIMEOUT_SECONDS)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"OpenAI API call failed: {e}")
        return None


def _call_anthropic(prompt: str, system_prompt: str = "") -> Optional[str]:
    """Call Anthropic API and return the response text."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=LLM_API_KEY, timeout=LLM_TIMEOUT_SECONDS)
        response = client.messages.create(
            model=LLM_MODEL or "claude-3-haiku-20240307",
            max_tokens=1000,
            system=system_prompt if system_prompt else "You are a resume screening assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        logger.warning(f"Anthropic API call failed: {e}")
        return None


def _call_gemini(prompt: str, system_prompt: str = "") -> Optional[str]:
    """Call Google Gemini API and return the response text."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=LLM_API_KEY)
        model = genai.GenerativeModel(
            model_name=LLM_MODEL or "gemini-2.0-flash",
            system_instruction=system_prompt if system_prompt else None,
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000,
            ),
        )
        return response.text
    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}")
        return None


def _call_stub(prompt: str, system_prompt: str = "") -> Optional[str]:
    """Stub LLM provider — returns None to trigger fallback."""
    logger.info("LLM stub called — falling back to rule-based logic")
    return None


PROVIDERS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
    "stub": _call_stub,
}


def call_llm(prompt: str, system_prompt: str = "") -> Optional[str]:
    """
    Provider-agnostic LLM call with rate limiting.
    Returns response text or None on failure.
    """
    if not LLM_ENABLED:
        logger.debug("LLM not enabled (no API key), using stub")
        return _call_stub(prompt, system_prompt)

    _wait_for_rate_limit()
    provider_func = PROVIDERS.get(LLM_PROVIDER, _call_stub)
    return provider_func(prompt, system_prompt)


def extract_candidate_with_llm(filename: str, raw_text: str) -> Optional[dict]:
    """
    Use LLM for structured extraction of candidate info.
    Returns a dict with extracted fields or None on failure.
    """
    system_prompt = (
        "You are a resume screening assistant. Extract structured information from the resume. "
        "Return valid JSON with the following fields: "
        '{"name": "...", "email": "...", "phone": "...", "github_url": "...", '
        '"skills": ["..."], "projects": ["..."], "project_descriptions": ["..."], '
        '"work_experience": ["..."], "education": ["..."]}'
    )

    prompt = f"Extract candidate information from this resume:\n\n{raw_text[:3000]}"

    response = call_llm(prompt, system_prompt)
    if not response:
        return None

    try:
        # Try to parse JSON from response
        # Handle markdown code blocks
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            response = "\n".join(json_lines)

        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        return None


def score_with_llm(
    candidate: Candidate,
    eligibility_info: str,
) -> Optional[tuple[float, ScoreBreakdown, list[str], list[str]]]:
    """
    Use LLM to judge candidate quality and provide score breakdown.
    Returns (total_score, breakdown, strengths, concerns) or None on failure.
    """
    system_prompt = (
        "You are a technical recruiter scoring candidates. Score the candidate 0-100 based on: "
        "AI/Agentic project depth (40 pts), Python & Backend (30 pts), "
        "Cloud/Deployment/Full-stack (15 pts), GitHub activity (10 pts), "
        "Engineering depth (5 pts). Deduct 5-15 pts for thin API wrappers or tutorial projects. "
        "Return valid JSON: "
        '{"total_score": N, "ai_project_depth": N, "python_backend": N, '
        '"cloud_fullstack": N, "github": N, "engineering_depth": N, '
        '"strengths": ["..."], "concerns": ["..."], "summary": "..."}'
    )

    prompt = (
        f"Candidate: {candidate.name or candidate.filename}\n"
        f"Skills: {', '.join(candidate.skills[:20])}\n"
        f"Projects: {'; '.join(candidate.projects[:5])}\n"
        f"Experience: {'; '.join(candidate.work_experience[:3])}\n"
        f"Education: {'; '.join(candidate.education[:2])}\n"
        f"Eligibility info: {eligibility_info}\n"
        f"GitHub: {candidate.github_url or 'Not provided'}\n"
    )

    response = call_llm(prompt, system_prompt)
    if not response:
        return None

    try:
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            response = "\n".join(json_lines)

        data = json.loads(response)
        breakdown = ScoreBreakdown(
            ai_project_depth=data.get("ai_project_depth", 0),
            python_backend=data.get("python_backend", 0),
            cloud_fullstack=data.get("cloud_fullstack", 0),
            github=data.get("github", 0),
            engineering_depth=data.get("engineering_depth", 0),
        )
        total = data.get("total_score", 0)
        strengths = data.get("strengths", [])
        concerns = data.get("concerns", [])
        return total, breakdown, strengths, concerns
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse LLM scoring response: {e}")
        return None
