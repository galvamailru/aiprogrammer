from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .models import BusinessTaskRequest
from .orchestrator import orchestrator
from .storage import storage

app = FastAPI(title="AI Programmer")

base_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/api/config")
async def get_config() -> dict:
    return {
        "local_project_dir": str(settings.local_project_path),
        "deploy_project_dir": settings.deploy_project_dir,
        "default_git_url": "",
        "auto_git_push": settings.auto_git_push,
        "auto_deploy": settings.auto_deploy,
        "max_fix_attempts": settings.max_fix_attempts,
    }


@app.post("/api/context/upload")
async def upload_context(file: UploadFile = File(...)) -> dict:
    if not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are allowed.")
    content = (await file.read()).decode("utf-8", errors="ignore")
    doc = storage.save_document(file.filename, content)
    return {"ok": True, "name": doc.name, "path": doc.path}


@app.get("/api/context")
async def list_context() -> list[dict]:
    docs = storage.list_documents()
    return [doc.model_dump() for doc in docs]


@app.post("/api/runs")
async def create_run(payload: BusinessTaskRequest) -> dict:
    task_text = payload.task_text.strip()
    if not task_text:
        raise HTTPException(status_code=400, detail="Task text cannot be empty.")
    run = orchestrator.start_run(
        task_text=task_text,
        git_url=payload.git_url,
        deploy_project_dir=payload.deploy_project_dir,
    )
    return {"ok": True, "run_id": run.run_id}


@app.get("/api/runs")
async def list_runs() -> list[dict]:
    return [run.model_dump(mode="json") for run in storage.list_runs()]


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    try:
        run = storage.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    return run.model_dump(mode="json")
