"""GitHub enrichment module — public GitHub API lookup for scoring signal."""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from src.config import (
    GITHUB_API_BASE,
    GITHUB_REPOS_MAX,
    GITHUB_RECENT_ACTIVITY_MAX,
    GITHUB_TIMEOUT_SECONDS,
    GITHUB_TOKEN,
)
from src.models import Candidate, GitHubEnrichment

logger = logging.getLogger(__name__)

# Simple in-memory cache for GitHub calls
_github_cache: dict[str, GitHubEnrichment] = {}


def _github_headers() -> dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def _get_json(url: str) -> Optional[dict | list]:
    """Make a GET request and return JSON, or None on failure."""
    try:
        resp = requests.get(url, headers=_github_headers(), timeout=GITHUB_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 403:
            logger.warning(f"GitHub rate limit hit for {url}")
        elif resp.status_code == 404:
            logger.info(f"GitHub resource not found: {url}")
        else:
            logger.warning(f"GitHub API returned {resp.status_code} for {url}")
    except requests.RequestException as e:
        logger.warning(f"GitHub API request failed: {e}")
    return None


def _get_recent_activity(username: str) -> tuple[float, str]:
    """
    Score recent public activity (0-5 points).
    Check recent events (pushes, PRs, issues).
    Returns (score, summary).
    """
    url = f"{GITHUB_API_BASE}/users/{username}/events/public"
    data = _get_json(url)
    if not data or not isinstance(data, list):
        return 0, "No recent activity data available"

    # Filter for recent events (within ~90 days)
    now = time.time()
    recent_events = []
    for event in data[:30]:  # Check up to 30 events
        created_at = event.get("created_at", "")
        if created_at:
            try:
                event_time = time.mktime(
                    time.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
                )
                if now - event_time < 90 * 24 * 3600:  # 90 days
                    recent_events.append(event)
            except (ValueError, OverflowError):
                continue

    if not recent_events:
        return 0, "No recent public activity (last 90 days)"

    # Score based on activity types
    activity_types = {}
    for event in recent_events:
        etype = event.get("type", "Unknown")
        activity_types[etype] = activity_types.get(etype, 0) + 1

    pushes = activity_types.get("PushEvent", 0)
    prs = activity_types.get("PullRequestEvent", 0)
    issues = activity_types.get("IssuesEvent", 0)
    reviews = activity_types.get("PullRequestReviewEvent", 0)

    score = 0
    if pushes >= 5:
        score += 2
    elif pushes >= 2:
        score += 1
    if prs >= 2:
        score += 2
    elif prs >= 1:
        score += 1
    if issues >= 2:
        score += 1
    if reviews >= 1:
        score += 1

    score = min(score, 5)
    summary = f"{len(recent_events)} events in 90 days (pushes:{pushes}, PRs:{prs}, issues:{issues})"
    return score, summary


def _get_repos_info(username: str) -> tuple[float, str, int, int]:
    """
    Score maintained/relevant repos (0-5 points).
    Returns (score, summary, recent_repos_count, maintained_repos_count).
    """
    url = f"{GITHUB_API_BASE}/users/{username}/repos?sort=updated&per_page=30"
    data = _get_json(url)
    if not data or not isinstance(data, list):
        return 0, "No repository data available", 0, 0

    now = time.time()
    recent_repos = 0
    maintained_repos = 0
    python_ai_repos = 0

    for repo in data[:GITHUB_REPOS_MAX * 2]:
        updated_at = repo.get("updated_at", "")
        pushed_at = repo.get("pushed_at", "")
        language = (repo.get("language") or "").lower()
        description = (repo.get("description") or "").lower()
        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)

        # Check if recently updated (within 180 days)
        if updated_at:
            try:
                update_time = time.mktime(
                    time.strptime(updated_at, "%Y-%m-%dT%H:%M:%SZ")
                )
                if now - update_time < 180 * 24 * 3600:
                    recent_repos += 1
            except (ValueError, OverflowError):
                pass

        # Check if maintained (has stars/forks or recently pushed)
        if stars > 0 or forks > 0:
            maintained_repos += 1

        # Check if Python/AI relevant
        if language == "python":
            ai_terms = ["ai", "ml", "llm", "rag", "agent", "nlp", "deep learning"]
            if any(term in description for term in ai_terms):
                python_ai_repos += 1

    recent_repos = min(recent_repos, GITHUB_RECENT_ACTIVITY_MAX)
    maintained_repos = min(maintained_repos, GITHUB_REPOS_MAX)

    # Score
    score = 0
    if recent_repos >= 3:
        score += 2
    elif recent_repos >= 1:
        score += 1
    if maintained_repos >= 3:
        score += 2
    elif maintained_repos >= 1:
        score += 1
    if python_ai_repos >= 1:
        score += 1

    score = min(score, 5)
    summary = (
        f"{recent_repos} recently updated repos, "
        f"{maintained_repos} maintained, "
        f"{python_ai_repos} Python/AI repos"
    )
    return score, summary, recent_repos, maintained_repos


def enrich_github(candidate: Candidate) -> GitHubEnrichment:
    """
    Look up GitHub activity for a candidate.
    Returns GitHubEnrichment with scores and summary.
    """
    username = candidate.github_username
    if not username:
        return GitHubEnrichment(
            status="no_profile",
            summary="No GitHub URL found in resume",
        )

    # Check cache
    if username in _github_cache:
        logger.info(f"GitHub cache hit for {username}")
        return _github_cache[username]

    logger.info(f"Looking up GitHub profile for {username}")

    # Get recent activity
    activity_score, activity_summary = _get_recent_activity(username)

    # Get repos info
    repos_score, repos_summary, recent_count, maintained_count = _get_repos_info(username)

    # Check if we got any data
    if activity_score == 0 and repos_score == 0 and "No" in activity_summary:
        # Might be rate limited or private
        if "rate limit" in activity_summary.lower():
            status = "failed"
            summary = "GitHub rate limit exceeded"
        else:
            status = "success"
            summary = "Limited public activity found"
    else:
        status = "success"
        summary = f"{activity_summary}; {repos_summary}"

    enrichment = GitHubEnrichment(
        status=status,
        recent_activity_score=activity_score,
        repos_score=repos_score,
        total_score=activity_score + repos_score,
        summary=summary,
        recent_repos_count=recent_count,
        maintained_repos_count=maintained_count,
    )

    _github_cache[username] = enrichment
    return enrichment


def clear_github_cache():
    """Clear the in-memory GitHub cache."""
    _github_cache.clear()
