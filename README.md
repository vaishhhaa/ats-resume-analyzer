# ATS Resume Analyzer | Dual Dashboard System

**An intelligent resume analysis and recruitment screening application built with Python, FastAPI, and Machine Learning.**

A dual-dashboard web application designed to help job seekers analyze their resumes and help recruiters screen, rank, and evaluate multiple candidate resumes using text-processing and similarity techniques.

[Features](#-key-features) · [Screenshots](#-application-screenshots) · [Tech Stack](#-technology-stack) · [Installation](#-installation-and-setup) · [Deployment](#-deployment)

---

## 📌 Project Overview

The **ATS Resume Analyzer** combines resume analysis and recruiter-oriented screening in one application.

The system uses Natural Language Processing (NLP), TF-IDF vectorization, and cosine similarity to analyze resume text and measure textual relevance. It provides separate dashboards for job seekers and recruiters, with the aim of making resume improvement and candidate screening easier to understand.

### 🎯 Project Objectives

- Analyze resume content and provide an ATS-oriented score.
- Identify missing or relevant keywords.
- Provide actionable resume improvement suggestions.
- Help recruiters organize and evaluate candidate resumes.
- Apply text-processing and machine-learning techniques to resume screening.
- Present analysis results through a user-friendly dual-dashboard interface.

---

## ✨ Key Features

### 1. 👤 User Dashboard — Resume Analysis

A dedicated dashboard for job seekers to review their resume and understand potential areas for improvement.

- **Resume Upload:** Submit a resume for analysis using the supported document formats.
- **ATS Score Analysis:** View the resume's calculated score according to the application's scoring logic.
- **Keyword Analysis:** Identify missing or relevant keywords in the analysis.
- **Resume Improvement Suggestions:** Receive suggestions to improve resume relevance and content.
- **Analysis Results:** View the output of resume processing in an understandable interface.

**Goal:** Help job seekers identify potential resume gaps before applying for a position.

![User Dashboard](screenshots/02-user-dashboard.png)

![Resume Analysis Results](screenshots/04-ats-score-results.png)

![Missing Keywords and Suggestions](screenshots/06-improvement-suggestions.png)

### 2. 🧑‍💼 Recruiter Dashboard — Candidate Screening

A separate interface focused on reviewing and evaluating candidate resumes.

- **Resume Collection Analysis:** Process candidate resumes through the supported analysis workflow.
- **Resume Ranking:** Rank resumes according to the implemented relevance-scoring logic.
- **Candidate Filtering:** Filter resume results using the application's available filtering options.
- **Similarity-Based Matching:** Use TF-IDF and cosine similarity to calculate textual relevance.
- **Candidate Review:** Present analysis results in a recruiter-oriented workflow.
- **Bulk Screening Workflow:** Designed to support analysis of large resume collections, including a 100+ resume workflow where supported by the available system resources.

**Goal:** Help recruiters organize resume screening and prioritize candidates for further review.

![Recruiter Dashboard](screenshots/07-recruiter-dashboard.png)

![Resume Ranking and Filtering](screenshots/08-resume-ranking.png)

![Candidate Analysis](screenshots/09-candidate-details.png)

> Note: Screenshots should show the actual implemented interface. Remove any feature or screenshot reference that does not match the current application.

---

## 🧠 How It Works

The system combines text extraction, preprocessing, vectorization, and similarity calculations to analyze resume content.

### User Workflow

1. Upload a supported resume document.
2. Extract text from the resume.
3. Process and analyze the extracted content.
4. Calculate the ATS-oriented score using the implemented scoring logic.
5. Display missing keywords and improvement suggestions.

### Recruiter Workflow

1. Provide candidate resumes through the supported workflow.
2. Extract and preprocess resume text.
3. Convert text into numerical representations using TF-IDF.
4. Calculate textual similarity using cosine similarity.
5. Rank and filter results using the implemented criteria.
6. Review the resulting candidate analysis.

### Machine Learning Approach

**TF-IDF (Term Frequency–Inverse Document Frequency)**

Converts text into numerical features based on term frequency and how informative terms are across the analyzed documents.

**Cosine Similarity**

Measures the similarity between numerical text vectors. The resulting score can support resume relevance ranking when the comparison inputs and scoring method are appropriate.

**NLP and Text Processing**

Libraries such as spaCy and NLTK can support text processing where configured in the application.

> An ATS-oriented score is not a universal ATS pass probability. Text similarity alone does not determine candidate suitability or guarantee that a resume will pass an employer's screening system.

---

## 🖥️ Application Screenshots

The following gallery documents the key interfaces of the application. Replace the image paths with the screenshots captured from your running project.

| Screen | Screenshot |
|---|---|
| Homepage | `screenshots/01-homepage.png` |
| User Dashboard | `screenshots/02-user-dashboard.png` |
| Resume Upload | `screenshots/03-resume-upload.png` |
| ATS Score Results | `screenshots/04-ats-score-results.png` |
| Missing Keywords | `screenshots/05-missing-keywords.png` |
| Improvement Suggestions | `screenshots/06-improvement-suggestions.png` |
| Recruiter Dashboard | `screenshots/07-recruiter-dashboard.png` |
| Resume Ranking | `screenshots/08-resume-ranking.png` |
| Candidate Details | `screenshots/09-candidate-details.png` |

Keep only screenshots for screens that actually exist in the project. GitHub will display committed images directly in this README when the paths match.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| FastAPI | Backend and API framework |
| Uvicorn | ASGI application server |
| Scikit-learn | Machine-learning and vectorization utilities |
| TF-IDF | Text feature extraction |
| Cosine Similarity | Textual relevance calculation |
| spaCy / NLTK | NLP processing where used |
| pdfplumber | PDF text extraction where used |
| python-docx | Word document processing where used |
| HTML, CSS, JavaScript or templates | Existing frontend implementation |
| Git and GitHub | Version control and source hosting |

---

## 🏗️ Project Structure

The exact directory names should match the existing source code. A typical layout for this application could look like this:

```text
ats-resume-analyzer/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── app/
│   ├── routes/
│   ├── services/
│   └── utils/
│
├── templates/
├── static/
├── models/
│
└── screenshots/
    ├── 01-homepage.png
    ├── 02-user-dashboard.png
    ├── 03-resume-upload.png
    ├── 04-ats-score-results.png
    ├── 05-missing-keywords.png
    ├── 06-improvement-suggestions.png
    ├── 07-recruiter-dashboard.png
    ├── 08-resume-ranking.png
    └── 09-candidate-details.png
```

The `app/`, `templates/`, `static/`, and `models/` directories are illustrative. Keep your existing folder structure if your code uses a different organization.

---

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system.
- Git installed.
- Access to the project repository.
- Required dependencies listed in `requirements.txt`.

### Step 1: Clone the Repository

```bash
git clone https://github.com/vaishhhaa/ats-resume-analyzer.git
cd ats-resume-analyzer
```

Replace the repository URL if your actual GitHub repository has a different name.

### Step 2: Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Run the Application

If the FastAPI application instance is named `app` inside `main.py`:

```bash
uvicorn main:app --reload
```

Use the correct module path if your entry point is located elsewhere.

### Step 5: Open the Application

- Local application: http://127.0.0.1:8000
- FastAPI interactive API documentation: http://127.0.0.1:8000/docs

The first URL displays your application interface if the frontend routes are configured. The second displays the API documentation generated by FastAPI.

---

## 🚀 Deployment

The application can be deployed as a Python web service on a hosting platform that supports FastAPI.

For example, with Render:

**Build command**

```bash
pip install -r requirements.txt
```

**Start command** — if the application entry point is `main.py`:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Update the command to match the actual entry point. Configure any required environment variables, model assets, and storage before deploying.

Deployment guide: https://render.com/docs/deploy-fastapi

**Important:** GitHub hosts the source code; it does not run the FastAPI backend by itself. A separate web service is required for a live application.

---

## 🔐 Privacy and Limitations

- Resume documents can contain sensitive personal information. Use anonymized sample resumes for public demonstrations.
- Never commit real candidate resumes, private documents, API keys, or credentials.
- ATS scores depend on the implemented scoring logic and input quality.
- Similarity scores measure textual relevance, not a candidate's overall professional ability.
- Resume parsing can be affected by document formatting, tables, columns, and scanned PDFs.
- Persistent storage, authentication, and data deletion should be documented according to the actual implementation.

---

## 🔮 Future Improvements

Potential enhancements, depending on the existing implementation, include:

- More robust resume parsing and validation.
- Configurable job-description matching.
- Explainable scoring and transparent ranking criteria.
- Secure authentication and role-based access.
- Persistent candidate management.
- Automated testing and improved deployment monitoring.

---

## 👩‍💻 Author

**Vaishnavi Chakor**

BCA Graduate | MCA Student

- GitHub: [@vaishhhaa](https://github.com/vaishhhaa)
- Portfolio: [View Portfolio](https://flourishing-buttercream-c2811d.netlify.app)
- LinkedIn: [Connect on LinkedIn](https://linkedin.com/in/vaishnavichakor)

---

## ⭐ Project Highlights

- Dual-dashboard application for job seekers and recruiters.
- Python backend using FastAPI and Uvicorn.
- Resume analysis using text-processing and machine-learning techniques.
- TF-IDF vectorization and cosine similarity for relevance analysis.
- Recruiter-focused resume ranking and screening workflow.

If you find this project useful, consider giving the repository a ⭐ on GitHub.
