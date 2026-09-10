"""Configuration module — weights, thresholds, model name, env-var driven."""

import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "resumes"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "results.json"

# --- Supported file extensions ---
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
PDF_EXTENSIONS = {".pdf"}

# --- Eligibility thresholds ---
MIN_PYTHON_EVIDENCE = True  # Must have Python skill/project evidence
MIN_AI_EVIDENCE = True  # Must have AI/LLM/RAG/agentic project evidence

# --- Scoring weights (must sum to 100 before penalties) ---
WEIGHT_AI_PROJECT_DEPTH = 40
WEIGHT_PYTHON_BACKEND = 30
WEIGHT_CLOUD_FULLSTACK = 15
WEIGHT_GITHUB = 10
WEIGHT_ENGINEERING_DEPTH = 5

# --- Penalty ranges ---
PENALTY_MIN = 5
PENALTY_MAX = 15
THIN_WRAPPER_PENALTY = 10  # for thin LLM/API wrappers
TUTORIAL_PENALTY = 8  # for tutorial-style projects with no implementation detail

# --- GitHub enrichment ---
GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_RECENT_ACTIVITY_MAX = 5
GITHUB_REPOS_MAX = 5
GITHUB_TIMEOUT_SECONDS = 10

# --- LLM settings (provider-agnostic) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai | anthropic | stub
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TIMEOUT_SECONDS = 30
LLM_ENABLED = bool(LLM_API_KEY)

# --- Concurrency ---
MAX_CONCURRENT_GITHUB = 5
MAX_CONCURRENT_LLM = 3
MAX_CONCURRENT_FILES = 10

# --- Skills keywords for extraction ---
PYTHON_KEYWORDS = [
    "python", "python3", "py", "cpython",
    "fastapi", "django", "flask", "uvicorn",
    "numpy", "pandas", "scipy", "scikit-learn", "sklearn",
    "pytest", "unittest", "pip", "conda", "poetry",
    "asyncio", "async/await", "celery", "sqlalchemy",
    "pydantic", "typer", "click",
]

AI_KEYWORDS = [
    "langchain", "langgraph", "llamaindex", "llama-index",
    "openai", "anthropic", "claude", "gpt", "llm", "large language model",
    "rag", "retrieval augmented generation", "retrieval-augmented",
    "vector database", "vector store", "embedding", "embeddings",
    "chromadb", "pinecone", "weaviate", "qdrant", "faiss", "milvus",
    "agent", "agentic", "multi-agent", "tool calling", "function calling",
    "google adk", "google agent development kit",
    "langsmith", "langfuse", "helicone", "evaluation", "eval pipeline",
    "hugging face", "huggingface", "transformers", "pytorch", "tensorflow",
    "stable diffusion", "midjourney", "dalle", "image generation",
    "speech recognition", "nlp", "natural language processing",
    "chatbot", "conversational ai", "fine-tuning", "fine tuning",
    "prompt engineering", "prompt engineering",
]

CLOUD_KEYWORDS = [
    "gcp", "google cloud", "aws", "amazon web services", "azure",
    "docker", "kubernetes", "k8s", "terraform", "ci/cd", "github actions",
    "heroku", "vercel", "netlify", "cloud functions", "lambda",
    "cloud run", "cloud build", "app engine",
]

ENGINEERING_DEPTH_KEYWORDS = [
    "testing", "unit test", "integration test", "tdd", "test driven",
    "architecture", "microservice", "microservices",
    "caching", "redis", "memcached",
    "queue", "rabbitmq", "kafka", "celery",
    "observability", "monitoring", "logging", "prometheus", "grafana",
    "concurrency", "async", "parallel",
    "ci/cd", "devops",
    "load balanc", "failover", "resilience",
]
