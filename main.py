import io
from pathlib import Path
from typing import List, Optional

import pdfplumber
from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .nlp import normalize_whitespace
from .scoring import ScoreResult, calculate_score

app = FastAPI(title="Resume ATS Score Analyzer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class ScoreResponse(BaseModel):
    ats_score: float
    matched_skills: list[str]
    missing_keywords: list[str]
    suggestions: list[str]
    breakdown: dict


class CandidateResult(BaseModel):
    filename: str
    ats_score: float
    matched_skills: list[str]
    missing_keywords: list[str]
    breakdown: dict


class BatchScoreResponse(BaseModel):
    candidates: list[CandidateResult]
    total_processed: int


def _read_pdf(file_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def _read_docx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as stream:
        doc = Document(stream)
    return "\n".join(para.text for para in doc.paragraphs)


def extract_text(upload: UploadFile) -> str:
    # Reset file pointer to beginning in case it was read before
    upload.file.seek(0)
    data = upload.file.read()
    upload.file.seek(0)  # Reset again for potential reuse
    
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    filename = upload.filename or ""
    if filename.lower().endswith(".pdf"):
        text = _read_pdf(data)
    elif filename.lower().endswith(".docx"):
        text = _read_docx(data)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type (use PDF or DOCX)")

    cleaned = normalize_whitespace(text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Could not extract text from file")
    return cleaned


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def serve_index():
    """
    Serve the frontend index.html so the app works as a single website on port 8000.
    """
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/recruiter")
def serve_recruiter():
    """
    Serve the recruiter dashboard page.
    """
    recruiter_path = FRONTEND_DIR / "recruiter.html"
    if not recruiter_path.exists():
        raise HTTPException(status_code=500, detail="Recruiter page not found")
    return FileResponse(recruiter_path)


@app.post("/score", response_model=ScoreResponse)
async def score_resume(
    job_text: Optional[str] = Form(None),
    job_file: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    resume_file: Optional[UploadFile] = File(None),
):
    # Extract job description
    if job_file:
        job_content = extract_text(job_file)
    elif job_text and job_text.strip():
        job_content = job_text.strip()
    else:
        raise HTTPException(status_code=400, detail="Provide job_text or job_file")

    # Extract resume
    if resume_text and resume_text.strip():
        resume_content = resume_text.strip()
    elif resume_file:
        resume_content = extract_text(resume_file)
    else:
        raise HTTPException(status_code=400, detail="Provide resume_text or resume_file")

    result: ScoreResult = calculate_score(resume_content, job_content)
    return ScoreResponse(**result.__dict__)


@app.post("/batch-score", response_model=BatchScoreResponse)
async def batch_score_resumes(
    job_text: Optional[str] = Form(None),
    job_file: Optional[UploadFile] = File(None),
    resume_files: List[UploadFile] = File(...),
):
    """
    Batch process multiple resumes against a job description.
    Returns scores for all candidates.
    """
    # Extract job description
    if job_file:
        job_content = extract_text(job_file)
    elif job_text and job_text.strip():
        job_content = job_text.strip()
    else:
        raise HTTPException(status_code=400, detail="Provide job_text or job_file")

    if not resume_files:
        raise HTTPException(status_code=400, detail="At least one resume file is required")

    candidates = []
    for resume_file in resume_files:
        try:
            resume_content = extract_text(resume_file)
            result: ScoreResult = calculate_score(resume_content, job_content)
            
            # Extract filename without extension as candidate identifier
            filename = resume_file.filename or "unknown"
            candidate_name = Path(filename).stem
            
            candidates.append(CandidateResult(
                filename=candidate_name,
                ats_score=result.ats_score,
                matched_skills=result.matched_skills,
                missing_keywords=result.missing_keywords,
                breakdown=result.breakdown,
            ))
        except Exception as e:
            # Skip files that fail to process, but continue with others
            continue

    return BatchScoreResponse(
        candidates=candidates,
        total_processed=len(candidates)
    )
