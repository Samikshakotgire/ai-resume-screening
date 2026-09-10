"""Extraction module — parse name, email, skills, projects, GitHub URL from raw text."""

from __future__ import annotations

import re
from typing import Optional

from src.config import AI_KEYWORDS, CLOUD_KEYWORDS, PYTHON_KEYWORDS, ENGINEERING_DEPTH_KEYWORDS
from src.models import Candidate


# --- Regex patterns ---
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE
)
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}"
)
GITHUB_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9\-_]+)", re.IGNORECASE
)
NAME_PATTERN = re.compile(
    r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    re.MULTILINE,
)

# Section headers
SKILLS_SECTION = re.compile(
    r"(?:skills?|technologies|technical\s+skills?|tech\s+stack|competencies)",
    re.IGNORECASE,
)
PROJECTS_SECTION = re.compile(
    r"(?:projects?|portfolio|personal\s+projects?)",
    re.IGNORECASE,
)
EXPERIENCE_SECTION = re.compile(
    r"(?:experience|work\s+experience|employment|internships?|positions?)",
    re.IGNORECASE,
)
EDUCATION_SECTION = re.compile(
    r"(?:education|academic|qualification|degree)",
    re.IGNORECASE,
)


def _extract_email(text: str) -> str:
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else ""


def _extract_phone(text: str) -> str:
    match = PHONE_PATTERN.search(text)
    return match.group(0).strip() if match else ""


def _extract_github(text: str) -> tuple[str, str]:
    """Return (github_url, github_username)."""
    match = GITHUB_PATTERN.search(text)
    if match:
        username = match.group(1)
        return f"https://github.com/{username}", username
    return "", ""


def _extract_name(text: str) -> str:
    """Heuristic name extraction — first plausible full name in the document."""
    # Try the first line first
    lines = text.strip().split("\n")
    for line in lines[:5]:
        line = line.strip()
        if not line or len(line) > 80:
            continue
        match = NAME_PATTERN.match(line)
        if match:
            return match.group(1).strip()
    # Fallback: first match in entire text
    match = NAME_PATTERN.search(text)
    return match.group(1).strip() if match else ""


def _extract_section(text: str, header_pattern: re.Pattern, max_lines: int = 30) -> str:
    """Extract text under a section header until the next header or max_lines."""
    lines = text.split("\n")
    capturing = False
    result_lines = []
    for line in lines:
        if header_pattern.search(line):
            capturing = True
            continue
        if capturing:
            stripped = line.strip()
            if not stripped:
                continue
            # Stop at next section header (all caps or markdown heading)
            if re.match(r"^[A-Z][A-Z\s]{3,}:?$", stripped) or stripped.startswith("#"):
                break
            result_lines.append(stripped)
            if len(result_lines) >= max_lines:
                break
    return "\n".join(result_lines)


def _extract_skills_from_section(section_text: str) -> list[str]:
    """Parse a comma/pipe/bullet-separated skills list."""
    # Split on common delimiters
    items = re.split(r"[,\|•\-\n]+", section_text)
    skills = []
    for item in items:
        item = item.strip().strip(":").strip()
        if item and len(item) < 60:
            skills.append(item)
    return skills


def _match_known_keywords(text_lower: str, keyword_list: list[str]) -> list[str]:
    """Return which known keywords appear in the text."""
    matched = []
    for kw in keyword_list:
        if kw.lower() in text_lower:
            matched.append(kw)
    return matched


def extract_candidate(filename: str, raw_text: str) -> Candidate:
    """Parse a raw resume text into a Candidate object."""
    candidate = Candidate(filename=filename, raw_text=raw_text)

    if not raw_text:
        return candidate

    text_lower = raw_text.lower()

    # Basic fields
    candidate.email = _extract_email(raw_text)
    candidate.phone = _extract_phone(raw_text)
    candidate.github_url, candidate.github_username = _extract_github(raw_text)
    candidate.name = _extract_name(raw_text)

    # Extract sections
    skills_section = _extract_section(raw_text, SKILLS_SECTION)
    projects_section = _extract_section(raw_text, PROJECTS_SECTION)
    experience_section = _extract_section(raw_text, EXPERIENCE_SECTION)
    education_section = _extract_section(raw_text, EDUCATION_SECTION)

    # Parse skills
    if skills_section:
        candidate.skills = _extract_skills_from_section(skills_section)

    # Parse projects (split by bullet or numbered items)
    if projects_section:
        raw_projects = re.split(r"(?:\d+[\.\)]\s*|\•\s*|\-\s*)", projects_section)
        candidate.projects = [p.strip() for p in raw_projects if p.strip()]

    # Work experience
    if experience_section:
        raw_exp = re.split(r"(?:\d+[\.\)]\s*|\•\s*|\-\s*)", experience_section)
        candidate.work_experience = [e.strip() for e in raw_exp if e.strip()]

    # Education
    if education_section:
        raw_edu = re.split(r"(?:\d+[\.\)]\s*|\•\s*|\-\s*)", education_section)
        candidate.education = [e.strip() for e in raw_edu if e.strip()]

    # Keyword matching across full text
    matched_python = _match_known_keywords(text_lower, PYTHON_KEYWORDS)
    matched_ai = _match_known_keywords(text_lower, AI_KEYWORDS)
    matched_cloud = _match_known_keywords(text_lower, CLOUD_KEYWORDS)
    matched_eng = _match_known_keywords(text_lower, ENGINEERING_DEPTH_KEYWORDS)

    # Add matched keywords to skills if not already present
    all_matched = list(set(matched_python + matched_ai + matched_cloud + matched_eng))
    for kw in all_matched:
        if kw.title() not in candidate.skills:
            candidate.skills.append(kw.title())

    return candidate
