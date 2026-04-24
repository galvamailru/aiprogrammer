from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .config import settings
from .models import AgentRun, ContextDocument, RunEvent, RunStatus

logger = logging.getLogger("aiprogrammer")


class InMemoryStorage:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: Dict[str, AgentRun] = {}
        self._docs: List[ContextDocument] = []
        settings.upload_path.mkdir(parents=True, exist_ok=True)
        settings.run_data_path.mkdir(parents=True, exist_ok=True)

    def save_document(self, file_name: str, content: str) -> ContextDocument:
        safe_name = f"{uuid.uuid4().hex}_{file_name}"
        file_path = settings.upload_path / safe_name
        file_path.write_text(content, encoding="utf-8")
        doc = ContextDocument(name=file_name, path=str(file_path), content=content)
        with self._lock:
            self._docs.append(doc)
        return doc

    def list_documents(self) -> List[ContextDocument]:
        with self._lock:
            return list(self._docs)

    def create_run(
        self,
        task_text: str,
        max_attempts: int,
        git_url: str,
        deploy_project_dir: str,
        architecture_spec: str = "",
        architect_prompt: str = "",
        use_repo_context: bool = True,
    ) -> AgentRun:
        run_id = uuid.uuid4().hex
        run = AgentRun(
            run_id=run_id,
            task_text=task_text,
            max_attempts=max_attempts,
            git_url=git_url,
            deploy_project_dir=deploy_project_dir,
            architecture_spec=architecture_spec,
            architect_prompt=architect_prompt,
            use_repo_context=use_repo_context,
        )
        with self._lock:
            self._runs[run_id] = run
        return run

    def update_status(self, run_id: str, status: RunStatus) -> None:
        with self._lock:
            run = self._runs[run_id]
            run.status = status
            run.updated_at = datetime.utcnow()

    def set_attempt(self, run_id: str, attempt: int) -> None:
        with self._lock:
            run = self._runs[run_id]
            run.current_attempt = attempt
            run.updated_at = datetime.utcnow()

    def add_event(self, run_id: str, stage: str, message: str, payload: dict | None = None) -> None:
        payload_data = payload or {}
        with self._lock:
            run = self._runs[run_id]
            run.events.append(RunEvent(stage=stage, message=message, payload=payload_data))
            run.updated_at = datetime.utcnow()
        logger.info("run=%s stage=%s message=%s payload=%s", run_id, stage, message, payload_data)

    def get_run(self, run_id: str) -> AgentRun:
        with self._lock:
            return self._runs[run_id].model_copy(deep=True)

    def list_runs(self) -> List[AgentRun]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._runs.values()]


storage = InMemoryStorage()
