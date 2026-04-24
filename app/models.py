from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    failed = "failed"
    completed = "completed"


class CommandResult(BaseModel):
    ok: bool
    command: str
    exit_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_sec: float = 0.0
    source: str = "local"


class RunEvent(BaseModel):
    ts: datetime = Field(default_factory=datetime.utcnow)
    stage: str
    message: str
    payload: dict = Field(default_factory=dict)


class ContextDocument(BaseModel):
    name: str
    path: str
    content: str


class BusinessTaskRequest(BaseModel):
    task_text: str
    git_url: str | None = None
    deploy_project_dir: str | None = None


class ArchitectDraftRequest(BaseModel):
    task_text: str
    architect_prompt: str | None = None


class ArchitectApproveRequest(BaseModel):
    task_text: str
    git_url: str | None = None
    deploy_project_dir: str | None = None
    architecture_spec: str
    architect_prompt: str | None = None


class GitAuthTestRequest(BaseModel):
    git_url: str


class CodeFileProposal(BaseModel):
    path: str
    content: str


class CodePlan(BaseModel):
    summary: str
    files: List[CodeFileProposal] = Field(default_factory=list)
    local_commands: List[str] = Field(default_factory=list)


class RepairAction(BaseModel):
    action_type: str
    target: str = ""
    command: str = ""
    file_path: str = ""
    find_text: str = ""
    replace_text: str = ""
    reason: str = ""


class RepairPlan(BaseModel):
    diagnosis: str
    confidence: float = 0.0
    actions: List[RepairAction] = Field(default_factory=list)
    expected_outcome: str = ""
    validation_steps: List[str] = Field(default_factory=list)


class AgentRun(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: RunStatus = RunStatus.pending
    task_text: str
    current_attempt: int = 0
    max_attempts: int = 3
    git_url: str = ""
    deploy_project_dir: str = ""
    architecture_spec: str = ""
    architect_prompt: str = ""
    events: List[RunEvent] = Field(default_factory=list)
