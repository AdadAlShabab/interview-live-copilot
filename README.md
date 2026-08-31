# Interview Copilot

A local FastAPI and Gemini-powered interview assistant. The resume source is the project-root `data/resume.md`; resume data, vector indexes, transcripts, and usage records stay under the configured local data directory.

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