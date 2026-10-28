"""
nlp.py — Text preprocessing and skill extraction
=================================================
Key design decisions:
  - extract_skills() ONLY matches against the curated TECH_SKILLS vocabulary.
    Generic spaCy NOUN fallback is intentionally removed — it was the source
    of noise like "base", "core", "date", "datum", "logic", "interaction".
  - preprocess() is used for TF-IDF cosine similarity (broad token coverage).
  - TECH_SKILLS is the single source of truth for what counts as a skill.
"""

import re
from functools import lru_cache
from typing import List, Set

import spacy
from spacy.language import Language
from spacy.lang.en.stop_words import STOP_WORDS


# ─────────────────────────────────────────────────────────────────────────────
# Extended stopwords
# ─────────────────────────────────────────────────────────────────────────────
STOPWORDS: Set[str] = set(STOP_WORDS) | {
    # JD boilerplate that is NOT a skill
    "experience", "work", "working", "ability", "strong", "good",
    "excellent", "knowledge", "understanding", "team", "teamwork",
    "collaborate", "communication", "skills", "skill", "proficiency",
    "proficient", "familiar", "familiarity", "required", "requirement",
    "responsibilities", "responsibility", "including", "plus", "preferred",
    "minimum", "year", "years", "month", "months", "day", "days",
    "hour", "hours", "man", "implementation", "deployment",
    "client", "clients", "business", "company", "organization", "role",
    "position", "candidate", "applicant", "application", "apply",
    "job", "opportunity", "need", "needs", "provide", "provides",
    "ensure", "maintain", "support", "use", "using", "used",
    "develop", "development", "build", "building", "create", "creating",
    "manage", "management", "process", "processes", "system", "systems",
    "solution", "solutions", "service", "services", "product", "products",
    "project", "projects", "task", "tasks", "timeline", "scope",
    "estimate", "effort", "total", "complete", "completion",
    "objective", "goal", "goals", "deliverable", "deliverables",
    "september", "october", "november", "december", "january",
    "february", "march", "april", "may", "june", "july", "august",
    "user", "users", "module", "modules", "feature", "features",
    "interface", "functionality", "workflow", "workflows",
    "documentation", "report", "reports", "feedback", "review",
    "approval", "approvals", "meeting", "meetings", "discussion",
    "proposal", "template", "templates", "format", "structure",
    "provided", "given", "based", "follow", "following",
    # Generic words that spaCy incorrectly tags as NOUN/PROPN
    "base", "core", "logic", "date", "datum", "interaction",
    "addition", "assistance", "assurance", "delivery", "efficiency",
    "control", "custom", "access", "validation", "configuration",
    "addition", "bug", "design", "backend", "frontend",
}


# ─────────────────────────────────────────────────────────────────────────────
# Noise pattern — reject these immediately
# ─────────────────────────────────────────────────────────────────────────────
_NOISE_PATTERN = re.compile(
    r"""
    ^\d+$                          |   # pure numbers
    ^\d+[-/]\d+                    |   # date-like  2024-01
    ^\d+\s*(man-hours?|hrs?|days?) |   # effort estimates
    ^[\W_]+$                           # punctuation only
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# TECH_SKILLS — curated vocabulary (the ONLY source for skill extraction)
# ─────────────────────────────────────────────────────────────────────────────
TECH_SKILLS: Set[str] = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
    "perl", "haskell", "elixir", "clojure", "dart",

    # Web / Frontend
    "html", "css", "react", "angular", "vue", "nextjs", "nuxtjs",
    "tailwind", "bootstrap", "sass", "scss", "webpack", "vite",
    "jquery", "svelte", "redux", "mobx", "graphql",

    # Backend / Frameworks
    "fastapi", "django", "flask", "node", "nodejs", "express",
    "spring", "springboot", "rails", "laravel", "nestjs",
    "rest", "restful", "api", "microservices", "grpc", "websocket",
    "oauth", "jwt",

    # Data / ML / AI
    "machine learning", "deep learning", "nlp",
    "natural language processing", "computer vision",
    "data analysis", "data science", "statistics",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    "llm", "bert", "transformers", "openai", "huggingface",
    "spark", "hadoop", "airflow", "etl", "tableau", "power bi",
    "looker", "dbt", "mlflow", "xgboost", "lightgbm",

    # Databases
    "sql", "mysql", "postgresql", "postgres", "mongodb", "redis",
    "elasticsearch", "cassandra", "sqlite", "oracle", "nosql",
    "firebase", "dynamodb", "mariadb", "neo4j", "influxdb",

    # Cloud / DevOps / Infra
    "aws", "azure", "gcp", "docker", "kubernetes", "k8s", "terraform",
    "ansible", "jenkins", "ci/cd", "github actions", "gitlab",
    "nginx", "linux", "bash", "shell", "helm", "prometheus", "grafana",

    # Tools / Platforms
    "git", "github", "gitlab", "jira", "confluence", "figma",
    "postman", "swagger", "celery", "kafka", "rabbitmq",
    "sonarqube", "datadog", "sentry", "elastic", "kibana",

    # Testing
    "pytest", "unittest", "jest", "cypress", "selenium",
    "playwright", "testing", "tdd", "bdd", "mocha", "chai",

    # Engineering concepts
    "agile", "scrum", "kanban", "devops", "devsecops",
    "authentication", "authorization", "encryption", "security",
    "performance", "optimization", "scalability", "caching",
    "logging", "monitoring", "debugging", "microservice",
    "serverless", "event-driven", "message queue",

    # Design / UX
    "ui", "ux", "wireframe", "prototype", "responsive",
    "accessibility",

    # Business tools
    "excel", "powerpoint", "salesforce", "sap", "erp", "crm",
}


# ─────────────────────────────────────────────────────────────────────────────
# spaCy model loader
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_nlp() -> Language:
    return spacy.load("en_core_web_sm")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> str:
    """
    Lowercase, lemmatize, strip stopwords/punctuation/numbers.
    Used for TF-IDF cosine similarity — casts a wider net intentionally.
    """
    if not text:
        return ""
    nlp = get_nlp()
    doc = nlp(text.lower())
    tokens: List[str] = []
    for token in doc:
        if token.is_stop or token.is_punct or token.like_num or token.is_space:
            continue
        lemma = token.lemma_.strip()
        if lemma and len(lemma) > 2 and lemma not in STOPWORDS:
            tokens.append(lemma)
    return " ".join(tokens)


def extract_skills(text: str) -> List[str]:
    """
    Extract skills by matching ONLY against the curated TECH_SKILLS set.

    NO generic spaCy NOUN fallback — that was the source of noise like
    'base', 'core', 'date', 'datum', 'logic', 'interaction', 'control'.

    Uses whole-word regex matching so 'r' doesn't match inside 'error',
    and 'c' doesn't match inside 'architecture'.
    """
    if not text:
        return []

    text_lower = text.lower()
    found: Set[str] = set()

    for skill in TECH_SKILLS:
        # Escape special chars (e.g. c++, c#, ci/cd) and match whole word
        escaped = re.escape(skill)
        # For skills ending in special chars (c++, c#), use lookahead
        pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill)

    return sorted(found)