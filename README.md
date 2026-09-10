# AI Resume Screening & Ranking System

A production-minded Python system that ingests PDF resumes, extracts candidate information, applies hard eligibility filters, scores candidates on a 100-point rubric, enriches with GitHub activity, and outputs ranked results.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Configure LLM provider
cp .env.example .env
# Edit .env to set LLM_PROVIDER, LLM_API_KEY, etc.

# (Optional) Set GitHub token for higher rate limits
export GITHUB_TOKEN=your_token_here
```

## Usage

```bash
# Basic run (rule-based only)
python main.py --input ./resumes --output ./output/results.json

# With LLM-assisted extraction and scoring
python main.py --input ./resumes --output ./output/results.json --use-llm

# Verbose logging
python main.py --input ./resumes --output ./output/results.json --verbose
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
project/
  resumes/                  # input PDFs
  src/
    ingestion.py            # load + extract raw text from each resume file
    extraction.py           # parse name/email/skills/projects/github url
    eligibility.py           # hard rule-based Python + AI/agentic filter
    scoring.py               # 100-point weighted scoring + penalties
    github_enrichment.py     # public GitHub API lookup, capped at 10 pts
    llm_adapter.py            # thin, swappable wrapper around LLM providers
    models.py                 # Pydantic schemas for candidate, eligibility, score
    config.py                 # weights, thresholds, model name, env-var driven
    pipeline.py                # orchestrates the full run, bounded concurrency
  main.py                      # CLI entry point
  tests/
    test_eligibility.py
    test_scoring.py
  README.md
  requirements.txt
  .env.example
```

## Design Decisions

### Filtering Strategy
Eligibility uses two hard, deterministic rules applied before any scoring:
1. **Python evidence**: Must appear as a genuine skill, project technology, work experience technology, or implementation language. Detected via keyword matching across skills, projects, work experience, and full resume text.
2. **AI/agentic evidence**: Must have at least one meaningful AI/LLM/RAG/agentic project, framework, or implementation. Checks for frameworks (LangChain, LlamaIndex, Google ADK), RAG signals (vector store, embeddings, semantic search), agentic signals (tool calling, multi-agent, orchestration), and implementation detail beyond keyword listing.

This two-gate approach ensures we only score candidates who can actually do the work, while being flexible about where evidence appears (skills list vs. project descriptions vs. work experience).

### Scoring Strategy
100-point weighted rubric applied only to eligible candidates:
- **AI/Agentic project depth (40 pts)**: Rewards real systems — agents, RAG, tools, retrieval, state, orchestration, evaluation. Checks for framework usage, RAG signals, agentic signals, evaluation metrics, and implementation detail.
- **Python & Backend engineering (30 pts)**: Rewards project/internship evidence over keyword-only skill lists. Checks for backend frameworks, database services, async patterns, testing, and evidence of Python usage in projects or work.
- **Cloud/Deployment/Full-stack (15 pts)**: Cloud platforms, DevOps tools, frontend as supporting signal. React/Next.js only counted within end-to-end systems.
- **GitHub activity (10 pts)**: Recent public activity (0-5) + maintained/relevant repos (0-5). Missing/private GitHub scores 0 here, never fails screening.
- **Engineering depth (5 pts)**: Testing, architecture, caching, queues, observability, concurrency signals.

**Penalties**: 5-15 pts deducted for thin LLM/API wrappers with no real workflow/retrieval/state/eval logic, and for tutorial-style projects with no implementation detail or ownership evidence.

### LLM Usage
- Optional, provider-agnostic via `llm_adapter.py`
- Hard eligibility checks remain deterministic (outside LLM)
- Structured output (JSON schema) for any LLM extraction or scoring
- One failed LLM call degrades gracefully to rule-based scoring
- Provider switched via `LLM_PROVIDER` env var (openai/anthropic/stub)

### GitHub Scoring
- Extracts username from resume via regex
- Unauthenticated public API calls (optional `GITHUB_TOKEN` for rate limits)
- Recent activity: events in last 90 days scored by type (pushes, PRs, issues)
- Repos: recently updated + maintained (stars/forks) + Python/AI relevant
- In-memory cache prevents duplicate API calls per run
- API failures logged, never crash the batch

## If I Had More Time

1. **Semantic similarity scoring**: Use embeddings to compare candidate project descriptions against ideal role requirements, enabling more nuanced matching beyond keyword detection.

2. **Multi-stage pipeline with human-in-the-loop**: Add a review stage where borderline candidates (e.g., scores 60-75) are flagged for human review, with the system providing its reasoning for each decision.

3. **Resume section detection improvements**: Replace regex-based section parsing with a trained NER model or layout-aware PDF parser (like pdfplumber with layout analysis) to handle diverse resume formats more reliably.

4. **Batch GitHub analysis with webhook integration**: Set up a lightweight webhook to receive GitHub push events for tracked candidates, building a real-time activity profile instead of relying on point-in-time API snapshots.
