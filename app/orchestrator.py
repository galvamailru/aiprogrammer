from __future__ import annotations

import asyncio
from pathlib import Path

from .config import settings
from .deepseek_client import DeepSeekClient
from .deployment import DeploymentService
from .models import AgentRun, CodePlan, RunStatus
from .storage import storage
from .terminal_runner import TerminalRunner


class AgentOrchestrator:
    def __init__(self) -> None:
        self.deepseek = DeepSeekClient()
        self.runner = TerminalRunner()
        self.deployment = DeploymentService(self.runner)

    def start_run(self, task_text: str, git_url: str | None = None, deploy_project_dir: str | None = None) -> AgentRun:
        effective_git_url = (git_url or "").strip()
        effective_deploy_dir = (deploy_project_dir or settings.deploy_project_dir).strip()
        run = storage.create_run(
            task_text=task_text,
            max_attempts=settings.max_fix_attempts,
            git_url=effective_git_url,
            deploy_project_dir=effective_deploy_dir,
        )
        storage.add_event(run.run_id, "intake", "Task accepted and queued.")
        storage.update_status(run.run_id, RunStatus.running)
        asyncio.create_task(self._run_pipeline(run.run_id))
        return run

    async def _run_pipeline(self, run_id: str) -> None:
        run = storage.get_run(run_id)
        docs = storage.list_documents()
        context_md = "\n\n".join([f"## {doc.name}\n{doc.content}" for doc in docs])[:50000]
        project_root = settings.local_project_path
        project_root.mkdir(parents=True, exist_ok=True)
        self._prepare_local_repo(project_root=project_root, git_url=run.git_url, run_id=run_id)

        for attempt in range(1, run.max_attempts + 1):
            storage.set_attempt(run_id, attempt)
            storage.add_event(run_id, "attempt", f"Attempt {attempt}/{run.max_attempts} started.")
            try:
                code_plan = await self.deepseek.build_code_plan(run.task_text, context_md)
                self._materialize_files(project_root, code_plan, run_id)
                self._run_local_commands(project_root, code_plan.local_commands, run_id)
                self._run_git_flow(project_root, run_id, attempt)

                if settings.auto_deploy:
                    deployed = await self._deploy_and_validate(run_id, run.git_url, run.deploy_project_dir)
                    if deployed:
                        storage.update_status(run_id, RunStatus.completed)
                        storage.add_event(run_id, "done", "Run completed successfully.")
                        return
                else:
                    storage.update_status(run_id, RunStatus.completed)
                    storage.add_event(run_id, "done", "Run completed (deploy disabled).")
                    return
            except Exception as exc:
                error_msg = str(exc)
                storage.add_event(run_id, "error", "Pipeline step failed.", {"error": error_msg})
                if attempt == run.max_attempts:
                    storage.update_status(run_id, RunStatus.failed)
                    storage.add_event(run_id, "failed", "Max attempts reached.")
                    return

                fix_hint = await self.deepseek.review_and_fix_hint(run.task_text, error_msg, context_md)
                storage.add_event(run_id, "review", "Generated fix hint for next attempt.", {"hint": fix_hint})

        storage.update_status(run_id, RunStatus.failed)
        storage.add_event(run_id, "failed", "Pipeline exited unexpectedly.")

    def _prepare_local_repo(self, project_root: Path, git_url: str, run_id: str) -> None:
        if not git_url:
            storage.add_event(run_id, "git", "git_url is empty, local repo preparation skipped.")
            return
        git_dir = project_root / ".git"
        parent = project_root.parent
        if not git_dir.exists():
            if any(project_root.iterdir()):
                storage.add_event(
                    run_id,
                    "git",
                    "Local project directory is not empty. Clone skipped, existing files will be used.",
                )
                return
            result = self.runner.run_local(
                ["git", "clone", git_url, str(project_root)],
                cwd=parent,
                timeout_sec=1200,
            )
            storage.add_event(run_id, "git", "Local repository clone executed.", payload=result.model_dump())
            if not result.ok:
                raise RuntimeError(f"Local clone failed: {result.stderr_tail}")
        else:
            result = self.runner.run_local(["git", "pull", "--ff-only"], cwd=project_root, timeout_sec=600)
            storage.add_event(run_id, "git", "Local repository pull executed.", payload=result.model_dump())
            if not result.ok:
                raise RuntimeError(f"Local pull failed: {result.stderr_tail}")

    def _materialize_files(self, root: Path, code_plan: CodePlan, run_id: str) -> None:
        storage.add_event(run_id, "plan", "Implementation plan generated.", {"summary": code_plan.summary})
        for item in code_plan.files:
            target = (root / item.path).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Refused to write outside project root: {item.path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item.content, encoding="utf-8")
            storage.add_event(run_id, "codegen", "File generated/updated.", {"path": str(target)})

    def _run_local_commands(self, root: Path, commands: list[str], run_id: str) -> None:
        if not commands:
            storage.add_event(run_id, "verify", "No local commands requested by plan.")
            return
        for raw in commands:
            command = ["python", "-c", f"print('Command placeholder: {raw}')"]
            result = self.runner.run_local(command, cwd=root, timeout_sec=600)
            storage.add_event(run_id, "verify", "Local command executed.", payload=result.model_dump())
            if not result.ok:
                raise RuntimeError(f"Local verification failed: {result.stderr_tail}")

    def _run_git_flow(self, root: Path, run_id: str, attempt: int) -> None:
        commands = [
            ["git", "add", "."],
            ["git", "commit", "-m", f"agent attempt {attempt}: auto implementation"],
        ]
        if settings.auto_git_push:
            commands.append(["git", "push", "origin", settings.local_git_branch])
        for command in commands:
            result = self.runner.run_local(command, cwd=root, timeout_sec=300)
            storage.add_event(run_id, "git", "Git command executed.", payload=result.model_dump())
            if command[1] == "commit" and result.exit_code != 0:
                # no-op commit is acceptable in repeated attempts
                continue
            if not result.ok:
                raise RuntimeError(f"Git flow failed: {result.stderr_tail}")

    async def _deploy_and_validate(self, run_id: str, git_url: str, deploy_project_dir: str) -> bool:
        if not git_url:
            raise RuntimeError("Deploy requires git_url from UI.")
        deploy_results = self.deployment.deploy_remote(git_url=git_url, deploy_project_dir=deploy_project_dir)
        for item in deploy_results:
            storage.add_event(run_id, "deploy", "Remote command executed.", payload=item.model_dump())
            if not item.ok:
                logs = self.deployment.fetch_remote_logs(deploy_project_dir=deploy_project_dir)
                storage.add_event(run_id, "deploy", "Deployment logs captured.", payload=logs.model_dump())
                raise RuntimeError(item.stderr_tail or item.stdout_tail or "Remote deploy failed.")

        health = self.deployment.healthcheck()
        storage.add_event(run_id, "health", "Healthcheck completed.", payload=health.model_dump())
        if not health.ok:
            logs = self.deployment.fetch_remote_logs(deploy_project_dir=deploy_project_dir)
            storage.add_event(run_id, "health", "Healthcheck failed, logs captured.", payload=logs.model_dump())
            raise RuntimeError(health.stderr_tail or "Healthcheck failed.")
        return True


orchestrator = AgentOrchestrator()
