# Interview Copilot

## Technical README Description

Interview Live Copilot is a Python-based interview assistance system designed to support candidates during live technical and behavioral interviews. The application ingests a candidate resume, extracts structured experience and story data, analyzes the target job description, and retrieves the most relevant evidence from a local vector store. When the interviewer asks a question, the system detects whether the input is a question, classifies its type, and generates a response grounded in the candidate’s actual experience and role requirements.

The backend is built with FastAPI and uses Google Gemini for structured response generation, while a local SQLite database stores interview sessions and transcript metadata. The system also maintains daily and per-session usage tracking to protect free-tier limits and prevent excessive API usage. The frontend is a lightweight desktop/browser interface that surfaces live transcript updates and suggested answers in near real time.

Key components include:
- Resume extraction and parsing via Gemini
- Job analysis and requirement matching
- Retrieval of relevant candidate evidence from a FAISS-backed local index
- Local question detection and classification heuristics
- Conversation memory summarization for follow-up context
- WebSocket-based real-time interview updates
- Desktop wrapper using PyWebView for local execution

## Run in a browser

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/ui/.

Copy `backend/.env.example` to `backend/.env` and set `GEMINI_API_KEY` to enable AI features. On the first run, `/api/resume/current` extracts the Markdown resume and stores the structured profile locally; later runs reuse the saved candidate profile. The first semantic retrieval run may download the Sentence Transformers model.

## Run as a desktop app

```powershell
..\.venv\Scripts\python.exe backend/desktop.py
```

PyWebView requires WebView2 on Windows. The browser mode is the simplest development workflow.

## Tests

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest -q
```

## API highlights

- `GET /api/resume/current` loads the persistent local resume profile.
- `POST /api/job/analyze` analyzes a job description.
- `POST /api/interview/answer` generates an evidence-grounded answer.
- `WS /api/interview/ws/{session_uuid}` runs the live transcript pipeline.
- `DELETE /api/privacy/candidate/{candidate_id}` deletes profile and vector data.
