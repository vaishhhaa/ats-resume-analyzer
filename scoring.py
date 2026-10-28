"""
scoring.py — ATS Resume Scoring Engine
=======================================

SCORING PHILOSOPHY
──────────────────
Raw TF-IDF cosine similarity between two short documents naturally produces
values of 0.02–0.15 even for strong matches. A doctor's CV matched against a
medical job posting may score 0.09 raw cosine — which maps to 9/100, which
is misleading and wrong.

This engine applies three calibration steps:

  1. SIGNAL RESCALING
     Each of the three raw signals (cosine, skill_cov, keyword_cov) is
     independently rescaled using observed realistic ranges:
       - cosine:      [0.0, 0.35]  → rescaled to [0, 1]
       - skill_cov:   [0.0, 1.0]   → used directly (already a ratio)
       - keyword_cov: [0.0, 1.0]   → used directly (already a ratio)

  2. WEIGHTED COMPOSITE
     score = (0.35 * cosine_scaled + 0.40 * skill_cov + 0.25 * kw_cov) * 100
     Skill coverage gets the highest weight because it's the most reliable signal.

  3. FLOOR ADJUSTMENT
     Any resume that has AT LEAST some skill or keyword overlap gets a minimum
     score of 45 — avoiding the demoralising "3%" display for a reasonable match.
     Perfect matches (all signals high) naturally score 75–95.

SCORE INTERPRETATION
────────────────────
  85–100 : Excellent — strong keyword, skill, and semantic alignment
  70–84  : Good      — most required skills present, minor gaps
  50–69  : Average   — partial match, clear improvement areas
  35–49  : Needs work — significant skill or keyword gaps
  0–34   : Poor      — major mismatch (different field, wrong domain)
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .nlp import STOPWORDS, TECH_SKILLS, extract_skills, preprocess


# ─────────────────────────────────────────────────────────────────────────────
# Skill Category Taxonomy  (for content-aware suggestions)
# ─────────────────────────────────────────────────────────────────────────────

SKILL_CATEGORIES: Dict[str, List[str]] = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#",
        "go", "rust", "ruby", "php", "swift", "kotlin", "scala", "r",
    ],
    "frontend": [
        "react", "angular", "vue", "nextjs", "html", "css", "tailwind",
        "bootstrap", "sass", "webpack", "vite",
    ],
    "backend": [
        "fastapi", "django", "flask", "nodejs", "express", "spring",
        "rails", "laravel", "graphql", "rest", "api", "microservices",
    ],
    "databases": [
        "sql", "mysql", "postgresql", "postgres", "mongodb", "redis",
        "elasticsearch", "cassandra", "sqlite", "oracle", "firebase",
        "dynamodb", "nosql",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "ansible", "jenkins", "ci/cd", "github actions", "nginx",
        "linux", "bash",
    ],
    "ml_ai": [
        "machine learning", "deep learning", "nlp", "tensorflow",
        "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "data analysis", "data science", "llm", "bert",
    ],
    "testing": [
        "pytest", "unittest", "jest", "cypress", "selenium",
        "testing", "tdd", "bdd",
    ],
    "tools": [
        "git", "github", "jira", "docker", "postman", "swagger",
        "kafka", "rabbitmq", "celery",
    ],
}

_SKILL_TO_CATEGORY: Dict[str, str] = {
    skill: cat
    for cat, skills in SKILL_CATEGORIES.items()
    for skill in skills
}

_CAT_LABELS: Dict[str, str] = {
    "programming_languages": "programming languages",
    "frontend":              "frontend technologies",
    "backend":               "backend / API frameworks",
    "databases":             "databases",
    "cloud_devops":          "cloud / DevOps tools",
    "ml_ai":                 "ML / AI libraries",
    "testing":               "testing tools",
    "tools":                 "developer tooling",
}

# Resume section detection
_SECTION_RE: Dict[str, re.Pattern] = {
    "summary":        re.compile(r"\b(summary|objective|profile|about me)\b", re.I),
    "experience":     re.compile(r"\b(experience|employment|work history|career)\b", re.I),
    "skills":         re.compile(r"\b(skills|technologies|competencies|tech stack|expertise)\b", re.I),
    "education":      re.compile(r"\b(education|degree|university|college|bachelor|master|phd)\b", re.I),
    "projects":       re.compile(r"\b(projects|portfolio|github|open.?source)\b", re.I),
    "certifications": re.compile(r"\b(certif|certified|certification|license|credential)\b", re.I),
    "metrics":        re.compile(
        r"(\d+\s*(%|percent|x\s+faster|x\s+improvement)|"
        r"\b(reduced|increased|improved|saved|grew|cut|boosted)\b.{0,40}\d+)", re.I),
    "action_verbs":   re.compile(
        r"\b(built|developed|designed|implemented|led|managed|optimized|created|"
        r"delivered|architected|automated|deployed|reduced|increased|improved|"
        r"integrated|migrated|scaled|refactored|launched|shipped)\b", re.I),
}

_JD_ACTION_VERBS_RE = re.compile(
    r"\b(built|developed|designed|implemented|led|managed|optimized|created|"
    r"delivered|architected|automated|deployed|reduced|increased|improved|"
    r"collaborated|maintained|integrated|migrated|scaled|monitored|reviewed|"
    r"coordinated|established|streamlined|mentored)\b", re.I,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    ats_score: float
    matched_skills: List[str]
    missing_keywords: List[str]
    suggestions: List[str]
    breakdown: Dict[str, float]


@dataclass
class _ResumeProfile:
    has_summary: bool
    has_experience: bool
    has_skills_section: bool
    has_education: bool
    has_projects: bool
    has_certifications: bool
    has_metrics: bool
    has_action_verbs: bool
    word_count: int
    missing_by_category: Dict[str, List[str]]
    matched_by_category: Dict[str, List[str]]
    weak_categories: List[str]
    strong_categories: List[str]
    jd_verbs_missing: List[str]
    cert_required_in_jd: bool
    jd_domain: str


# ─────────────────────────────────────────────────────────────────────────────
# Signal 1 — TF-IDF Cosine Similarity (with calibrated rescaling)
# ─────────────────────────────────────────────────────────────────────────────

# Realistic observed range for cosine similarity between two
# related-but-distinct documents (resume vs JD).
# Values above COSINE_MAX are treated as 1.0 (perfect).
_COSINE_REALISTIC_MAX = 0.30   # raw cosine ≥ 0.30 → treated as 100% signal
_COSINE_REALISTIC_MIN = 0.01   # raw cosine ≤ 0.01 → treated as 0% signal


def _tfidf_cosine_raw(resume_text: str, job_text: str) -> float:
    """Return raw TF-IDF cosine similarity (0.0 – 1.0)."""
    cleaned_resume = preprocess(resume_text)
    cleaned_job    = preprocess(job_text)

    if not cleaned_resume or not cleaned_job:
        return 0.0
    if len(cleaned_resume) < 15 or len(cleaned_job) < 15:
        return 0.0

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    try:
        tfidf = vectorizer.fit_transform([cleaned_resume, cleaned_job])
        sim   = cosine_similarity(tfidf[0:1], tfidf[1:2])
        return float(sim[0, 0]) if sim.size else 0.0
    except ValueError:
        return 0.0


def _rescale_cosine(raw: float) -> float:
    """
    Rescale raw cosine from [_COSINE_REALISTIC_MIN, _COSINE_REALISTIC_MAX]
    to [0.0, 1.0] so it contributes meaningfully to the composite score.
    """
    if raw <= _COSINE_REALISTIC_MIN:
        return 0.0
    if raw >= _COSINE_REALISTIC_MAX:
        return 1.0
    return (raw - _COSINE_REALISTIC_MIN) / (_COSINE_REALISTIC_MAX - _COSINE_REALISTIC_MIN)


# ─────────────────────────────────────────────────────────────────────────────
# Signal 2 — Skill Overlap
# ─────────────────────────────────────────────────────────────────────────────

def _skill_overlap(
    resume_text: str,
    job_text: str,
) -> Tuple[List[str], List[str], float]:
    """
    Overlap of TECH_SKILLS between resume and JD.
    Returns (matched, missing, coverage_ratio 0–1).
    """
    resume_skills: Set[str] = set(extract_skills(resume_text))
    job_skills:    Set[str] = set(extract_skills(job_text))

    if not job_skills:
        return [], [], 0.0

    matched  = sorted(resume_skills & job_skills)
    missing  = sorted(job_skills - resume_skills)
    coverage = len(matched) / len(job_skills)
    return matched, missing, coverage


# ─────────────────────────────────────────────────────────────────────────────
# Signal 3 — Keyword Coverage
# ─────────────────────────────────────────────────────────────────────────────

def _extract_top_keywords(text: str, top_n: int = 35) -> Set[str]:
    """
    Extract top-N meaningful keywords from text using TF-IDF.
    Only keeps words that are in TECH_SKILLS OR are ≥4 chars and not stopwords.
    """
    cleaned = preprocess(text)
    if not cleaned or len(cleaned) < 15:
        return set()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1), min_df=1,
        stop_words="english", max_features=200,
    )
    try:
        tfidf   = vectorizer.fit_transform([cleaned])
        names   = vectorizer.get_feature_names_out()
        scores  = tfidf.toarray()[0]
        ranked  = sorted(zip(names, scores), key=lambda x: -x[1])

        top_kws: Set[str] = set()
        for word, _ in ranked:
            if len(top_kws) >= top_n:
                break
            if word in TECH_SKILLS or (
                len(word) >= 4
                and word not in STOPWORDS
                and re.search(r"[a-z]{3}", word)
                # Reject generic words that slipped through preprocessing
                and word not in {
                    "base", "core", "date", "datum", "logic", "control",
                    "custom", "access", "design", "deliver", "deliver",
                    "addition", "assistance", "assurance", "efficiency",
                    "interaction", "validation", "configuration", "backend",
                    "frontend", "function", "method", "class", "object",
                    "variable", "instance", "type", "value", "result",
                    "output", "input", "data", "file", "list", "item",
                    "error", "exception", "request", "response", "message",
                }
            ):
                top_kws.add(word)
        return top_kws
    except ValueError:
        return set()


def _keyword_coverage(
    resume_text: str,
    job_text: str,
) -> Tuple[List[str], List[str], float]:
    """
    What fraction of the JD's important keywords appear in the resume?
    Returns (matched_kws, missing_kws, ratio 0–1).
    """
    job_keywords = _extract_top_keywords(job_text, top_n=35)

    if not job_keywords:
        # Hard fallback: all preprocessed JD tokens ≥ 5 chars
        job_clean    = preprocess(job_text)
        job_keywords = {
            w for w in job_clean.split()
            if len(w) >= 5 and w not in STOPWORDS
        }

    if not job_keywords:
        return [], [], 0.0

    resume_clean  = preprocess(resume_text)
    resume_tokens = set(resume_clean.split())

    matched  = sorted(resume_tokens & job_keywords)
    missing  = sorted(job_keywords - resume_tokens)
    coverage = len(matched) / len(job_keywords)
    return matched, missing, coverage


# ─────────────────────────────────────────────────────────────────────────────
# Composite Score (with floor calibration)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ats_score(
    cosine_raw: float,
    skill_cov: float,
    kw_cov: float,
    n_matched_skills: int,
) -> float:
    """
    Compute final ATS score (0–100) with calibration.

    Step 1: Rescale cosine to its realistic range.
    Step 2: Weighted composite.
    Step 3: Apply soft floor so any resume with real skill matches
            always scores ≥ 45 (avoids misleading single-digit display).
    """
    cosine_scaled = _rescale_cosine(cosine_raw)

    # Weights: skill coverage most reliable → highest weight
    raw = (
        0.35 * cosine_scaled +
        0.40 * skill_cov     +
        0.25 * kw_cov
    ) * 100

    score = float(np.clip(raw, 0.0, 100.0))

    # Soft floor: if the resume has at least some real skill matches,
    # it should never show below 45 — that's genuinely demoralising and wrong.
    if n_matched_skills >= 3 and score < 45.0:
        # Blend up proportionally based on skill count (more matches → higher floor)
        floor = min(45.0 + (n_matched_skills - 3) * 2.0, 60.0)
        score = floor + (score / 45.0) * (floor * 0.3)   # smooth blend
        score = float(np.clip(score, 0.0, 100.0))

    elif n_matched_skills >= 1 and score < 35.0:
        # At least 1 skill match → minimum 35
        score = max(score, 35.0)

    return round(score, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Resume Profile Builder
# ─────────────────────────────────────────────────────────────────────────────

def _detect_domain(job_text: str) -> str:
    t = job_text.lower()
    if re.search(r"\b(machine learning|deep learning|data science|nlp|ai model)\b", t):
        return "ML / AI"
    if re.search(r"\b(frontend|react|angular|vue|ui/ux|css|html)\b", t):
        return "frontend"
    if re.search(r"\b(backend|api|rest|microservices|server-side)\b", t):
        return "backend"
    if re.search(r"\b(devops|kubernetes|docker|ci/cd|infrastructure|cloud)\b", t):
        return "DevOps / Cloud"
    if re.search(r"\b(fullstack|full.stack|full stack)\b", t):
        return "full-stack"
    if re.search(r"\b(android|ios|mobile|flutter|react native)\b", t):
        return "mobile"
    if re.search(r"\b(security|pentest|vulnerability|cybersecurity)\b", t):
        return "cybersecurity"
    if re.search(r"\b(data engineer|spark|airflow|etl|pipeline|warehouse)\b", t):
        return "data engineering"
    return "software engineering"


def _build_profile(
    resume_text: str,
    job_text: str,
    matched_skills: List[str],
    missing_skills: List[str],
) -> _ResumeProfile:
    sections = {k: bool(p.search(resume_text)) for k, p in _SECTION_RE.items()}

    missing_by_cat: Dict[str, List[str]] = {}
    matched_by_cat: Dict[str, List[str]] = {}
    for s in missing_skills:
        cat = _SKILL_TO_CATEGORY.get(s)
        if cat:
            missing_by_cat.setdefault(cat, []).append(s)
    for s in matched_skills:
        cat = _SKILL_TO_CATEGORY.get(s)
        if cat:
            matched_by_cat.setdefault(cat, []).append(s)

    all_cats = set(missing_by_cat) | set(matched_by_cat)
    weak_cats, strong_cats = [], []
    for cat in all_cats:
        total   = len(missing_by_cat.get(cat, [])) + len(matched_by_cat.get(cat, []))
        matched = len(matched_by_cat.get(cat, []))
        if total == 0:
            continue
        ratio = matched / total
        if ratio < 0.5:
            weak_cats.append(cat)
        elif ratio >= 0.8:
            strong_cats.append(cat)

    jd_verbs   = {v.lower() for v in _JD_ACTION_VERBS_RE.findall(job_text)}
    res_tokens = set(resume_text.lower().split())
    missing_jd_verbs = sorted(jd_verbs - res_tokens)[:4]

    cert_required = bool(re.search(
        r"\b(certified|certification|certificate|aws certified|"
        r"cka|ckad|pmp|scrum master|cissp|comptia|azure certified)\b",
        job_text, re.I,
    ))

    return _ResumeProfile(
        has_summary         = sections["summary"],
        has_experience      = sections["experience"],
        has_skills_section  = sections["skills"],
        has_education       = sections["education"],
        has_projects        = sections["projects"],
        has_certifications  = sections["certifications"],
        has_metrics         = sections["metrics"],
        has_action_verbs    = sections["action_verbs"],
        word_count          = len(resume_text.split()),
        missing_by_category = missing_by_cat,
        matched_by_category = matched_by_cat,
        weak_categories     = weak_cats,
        strong_categories   = strong_cats,
        jd_verbs_missing    = missing_jd_verbs,
        cert_required_in_jd = cert_required,
        jd_domain           = _detect_domain(job_text),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Content-Aware Suggestion Engine
# ─────────────────────────────────────────────────────────────────────────────

def _generate_suggestions(
    profile: _ResumeProfile,
    ats_score: float,
    cosine_raw: float,
    skill_cov: float,
    keyword_cov: float,
    missing_keywords: List[str],
    job_text: str,
    resume_text: str,
) -> List[str]:
    suggestions: List[str] = []

    # S1 — Weakest skill category (most missing skills)
    if profile.weak_categories:
        top_cat = max(
            profile.weak_categories,
            key=lambda c: len(profile.missing_by_category.get(c, [])),
        )
        missing_in_top = profile.missing_by_category.get(top_cat, [])[:5]
        cat_label      = _CAT_LABELS.get(top_cat, top_cat)
        if missing_in_top:
            skill_str = ", ".join(f'"{s}"' for s in missing_in_top)
            already   = profile.matched_by_category.get(top_cat, [])
            if already:
                already_str = ", ".join(already[:3])
                suggestions.append(
                    f"You have some {cat_label} experience ({already_str}) but the role "
                    f"also requires {skill_str}. Add these to your Skills section."
                )
            else:
                suggestions.append(
                    f"Your resume has no {cat_label} coverage, but the job requires "
                    f"{skill_str}. List even beginner-level experience with these."
                )

    # S2 — Missing testing/tools (often overlooked)
    specific_missing = (
        profile.missing_by_category.get("testing", []) +
        profile.missing_by_category.get("tools", [])
    )[:3]
    if specific_missing and len(suggestions) < 5:
        st_str = ", ".join(f'"{s}"' for s in specific_missing)
        suggestions.append(
            f"The role expects {st_str}. Mention these in a projects or "
            f"experience bullet — even basic usage in a side project counts."
        )

    # S3 — No summary section
    if not profile.has_summary and len(suggestions) < 5:
        suggestions.append(
            f"Add a professional Summary at the top — 2–3 sentences naming "
            f"the role (e.g. '{profile.jd_domain} Developer') and your key strengths. "
            f"ATS parsers scan this section first."
        )

    # S4 — No skills section
    if not profile.has_skills_section and len(suggestions) < 5:
        cat_hint = (_CAT_LABELS.get(profile.weak_categories[0], "technical")
                    if profile.weak_categories else "technical")
        suggestions.append(
            f"Create a dedicated Skills section listing your {cat_hint} stack as "
            f"plain keywords — ATS engines specifically parse this section."
        )

    # S5 — No metrics in experience
    if not profile.has_metrics and profile.has_experience and len(suggestions) < 5:
        domain = profile.jd_domain
        if "ML" in domain or "AI" in domain:
            ex = "e.g. 'achieved 93% F1-score', 'cut inference latency by 40%'"
        elif "backend" in domain:
            ex = "e.g. 'reduced API latency from 800ms to 120ms', 'served 50k req/day'"
        elif "frontend" in domain:
            ex = "e.g. 'improved Lighthouse score from 62→94', 'cut bundle size 35%'"
        elif "DevOps" in domain or "Cloud" in domain:
            ex = "e.g. 'reduced deploy time by 70%', 'maintained 99.95% uptime'"
        else:
            ex = "e.g. 'delivered feature 3 weeks early', 'reduced bug reports by 45%'"
        suggestions.append(
            f"No quantified results found in your experience. Add numbers "
            f"({ex}) — metrics make your impact concrete and rank higher in ATS."
        )

    # S6 — JD verbs missing from resume
    if profile.jd_verbs_missing and len(suggestions) < 5:
        verb_str = ", ".join(f'"{v}"' for v in profile.jd_verbs_missing)
        suggestions.append(
            f"The job description uses verbs ({verb_str}) absent from your resume. "
            f"Start 2+ bullet points with these verbs to signal role alignment."
        )

    # S7 — Semantic mismatch
    if cosine_raw < 0.05 and len(suggestions) < 5:
        suggestions.append(
            f"Very low semantic overlap with this {profile.jd_domain} role. "
            f"Rewrite your summary and top 3 bullets using the exact phrasing "
            f"from the job description rather than synonyms."
        )
    elif cosine_raw < 0.12 and len(suggestions) < 5:
        suggestions.append(
            f"Moderate language mismatch for this {profile.jd_domain} role. "
            f"Find the 4–5 most repeated phrases in the JD and weave them "
            f"naturally into your existing bullet points."
        )

    # S8 — Resume too thin
    if profile.word_count < 280 and len(suggestions) < 5:
        suggestions.append(
            f"Your resume is only ~{profile.word_count} words (ideal: 400–700). "
            f"Expand bullets with: tech stack used, team size, and measurable outcome."
        )

    # S9 — No projects and no experience
    if not profile.has_projects and not profile.has_experience and len(suggestions) < 5:
        suggestions.append(
            f"No Experience or Projects section found. Add 2–3 projects using "
            f"the {profile.jd_domain} stack with a GitHub link and problem description."
        )

    # S10 — Cert gap
    if profile.cert_required_in_jd and not profile.has_certifications and len(suggestions) < 5:
        suggestions.append(
            f"This role mentions certifications. Add a Certifications section if you "
            f"hold any — or note the most relevant cert to pursue for {profile.jd_domain}."
        )

    # S11 — Keyword density fallback
    if keyword_cov < 0.35 and len(suggestions) < 5:
        clean_kws = [k for k in missing_keywords if k in TECH_SKILLS][:5]
        if clean_kws:
            kw_str = ", ".join(f'"{k}"' for k in clean_kws)
            suggestions.append(
                f"Low keyword density. Integrate {kw_str} into your experience "
                f"descriptions — show HOW you used them, not just that you know them."
            )

    # Fallback
    if not suggestions:
        suggestions.append(
            "Good baseline match. Tailor each bullet point to directly address "
            "a specific responsibility in the job description to push above 80."
        )

    return suggestions[:5]


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def calculate_score(resume_text: str, job_text: str) -> ScoreResult:
    resume_text = (resume_text or "").strip()
    job_text    = (job_text    or "").strip()

    # ── Three raw signals ──────────────────────────────────────────────────
    cosine_raw                                = _tfidf_cosine_raw(resume_text, job_text)
    matched_skills, missing_skills, skill_cov = _skill_overlap(resume_text, job_text)
    matched_kws,    missing_kws,    kw_cov    = _keyword_coverage(resume_text, job_text)

    # ── Calibrated composite score ─────────────────────────────────────────
    ats_score = _compute_ats_score(
        cosine_raw        = cosine_raw,
        skill_cov         = skill_cov,
        kw_cov            = kw_cov,
        n_matched_skills  = len(matched_skills),
    )

    # ── Breakdown — show rescaled cosine as percentage (not raw 0.03) ──────
    cosine_display = round(_rescale_cosine(cosine_raw), 4)

    # ── Merged matched / missing ───────────────────────────────────────────
    # Only show TECH_SKILLS in matched (no noise)
    all_matched = sorted(
        {s for s in (set(matched_skills) | set(matched_kws)) if s in TECH_SKILLS}
    )

    # Missing: only TECH_SKILLS, de-duped, capped at 15
    combined_missing = list(dict.fromkeys(missing_skills + missing_kws))
    all_missing = [k for k in combined_missing if k in TECH_SKILLS][:15]

    # ── Resume profile for suggestions ────────────────────────────────────
    profile = _build_profile(
        resume_text    = resume_text,
        job_text       = job_text,
        matched_skills = matched_skills,
        missing_skills = missing_skills,
    )

    # ── Content-aware suggestions ─────────────────────────────────────────
    suggestions = _generate_suggestions(
        profile          = profile,
        ats_score        = ats_score,
        cosine_raw       = cosine_raw,
        skill_cov        = skill_cov,
        keyword_cov      = kw_cov,
        missing_keywords = missing_kws,
        job_text         = job_text,
        resume_text      = resume_text,
    )

    return ScoreResult(
        ats_score        = ats_score,
        matched_skills   = all_matched,
        missing_keywords = all_missing,
        suggestions      = suggestions,
        breakdown        = {
            # Display rescaled values so the UI shows meaningful percentages
            "cosine_similarity": cosine_display,
            "skill_coverage":    round(skill_cov, 4),
            "keyword_coverage":  round(kw_cov,    4),
        },
    )  