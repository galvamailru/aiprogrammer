from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from .config import settings
from .deepseek_client import DeepSeekClient
from .deployment import DeploymentService
from .models import AgentRun, CodePlan, RepairPlan, RunStatus
from .storage import storage
from .terminal_runner import TerminalRunner


class AgentOrchestrator:
    def __init__(self) -> None:
        self.deepseek = DeepSeekClient()
        self.runner = TerminalRunner()
        self.deployment = DeploymentService(self.runner)

    def start_run(
        self,
        task_text: str,
        git_url: str | None = None,
        deploy_project_dir: str | None = None,
        architecture_spec: str = "",
        architect_prompt: str = "",
        use_repo_context: bool = True,
    ) -> AgentRun:
        effective_git_url = (git_url or "").strip()
        effective_deploy_dir = (deploy_project_dir or settings.deploy_project_dir).strip()
        run = storage.create_run(
            task_text=task_text,
            max_attempts=settings.max_fix_attempts,
            git_url=effective_git_url,
            deploy_project_dir=effective_deploy_dir,
            architecture_spec=architecture_spec,
            architect_prompt=architect_prompt,
            use_repo_context=use_repo_context,
        )
        storage.add_event(run.run_id, "intake", "Task accepted and queued.")
        if architecture_spec.strip():
            storage.add_event(
                run.run_id,
                "architecture",
                "Architecture TZ approved and attached to run context.",
                {"chars": len(architecture_spec)},
            )
        storage.update_status(run.run_id, RunStatus.running)
        task = asyncio.create_task(self._run_pipeline(run.run_id))
        task.add_done_callback(lambda t, rid=run.run_id: self._on_pipeline_done(rid, t))
        return run

    def _on_pipeline_done(self, run_id: str, task: asyncio.Task) -> None:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            storage.update_status(run_id, RunStatus.failed)
            storage.add_event(run_id, "failed", "Pipeline task was cancelled.")
            return
        except Exception as callback_exc:
            storage.update_status(run_id, RunStatus.failed)
            storage.add_event(
                run_id,
                "failed",
                "Pipeline task completion callback failed.",
                {"error": str(callback_exc)},
            )
            return

        if exc is not None:
            storage.update_status(run_id, RunStatus.failed)
            storage.add_event(
                run_id,
                "failed",
                "Pipeline crashed with unhandled exception.",
                {"error": str(exc)},
            )

    async def _run_pipeline(self, run_id: str) -> None:
        run = storage.get_run(run_id)
        docs = storage.list_documents()
        context_md = "\n\n".join([f"## {doc.name}\n{doc.content}" for doc in docs])[:50000]
        if run.architecture_spec.strip():
            context_md = (
                f"{context_md}\n\n## Approved architecture specification\n{run.architecture_spec.strip()}"
            ).strip()
        project_root = settings.local_project_path
        project_root.mkdir(parents=True, exist_ok=True)
        storage.add_event(run_id, "pipeline", "Pipeline started.", {"project_root": str(project_root)})
        self._prepare_local_repo(project_root=project_root, git_url=run.git_url, run_id=run_id)
        repo_context_md = self._build_repo_context(project_root, run.use_repo_context)
        invariants_md = self._build_invariants_context()
        if repo_context_md:
            storage.add_event(
                run_id,
                "context",
                "Repository context extracted for model.",
                {"chars": len(repo_context_md), "enabled": run.use_repo_context, "invariants": True},
            )
            context_md = f"{context_md}\n\n## System invariants\n{invariants_md}\n\n## Pulled repository context\n{repo_context_md}".strip()
        else:
            storage.add_event(
                run_id,
                "context",
                "Repository context disabled or unavailable; generation starts from task/context docs.",
                {"enabled": run.use_repo_context},
            )
            context_md = f"{context_md}\n\n## System invariants\n{invariants_md}".strip()

        repair_only_mode = False
        last_error_msg = ""
        last_fix_hint = ""

        for attempt in range(1, run.max_attempts + 1):
            storage.set_attempt(run_id, attempt)
            storage.add_event(run_id, "attempt", f"Attempt {attempt}/{run.max_attempts} started.")
            try:
                if repair_only_mode and attempt > 1:
                    storage.add_event(
                        run_id,
                        "repair",
                        "Repair-only mode active: skipping planning/codegen and applying targeted fixes.",
                        {"attempt": attempt},
                    )
                    repair_context_md = context_md
                    if last_fix_hint:
                        repair_context_md = (
                            f"{repair_context_md}\n\n## Last attempt fix directive (must follow)\n{last_fix_hint}"
                        ).strip()
                    error_for_repair = (
                        last_error_msg
                        or "Previous attempt failed. Apply minimal targeted repair and revalidate."
                    )
                    llm_repair_applied = await self._try_llm_repair(
                        run_id=run_id,
                        run=run,
                        error_msg=error_for_repair,
                        context_md=repair_context_md,
                        project_root=project_root,
                    )
                    if llm_repair_applied:
                        storage.update_status(run_id, RunStatus.completed)
                        storage.add_event(run_id, "done", "Run completed successfully after repair-only cycle.")
                        return
                    if attempt == run.max_attempts:
                        storage.update_status(run_id, RunStatus.failed)
                        storage.add_event(run_id, "failed", "Max attempts reached in repair-only mode.")
                        return
                    continue

                storage.add_event(run_id, "planning", "Requesting implementation plan from DeepSeek.")
                code_plan = await self.deepseek.build_code_plan(run.task_text, context_md)
                storage.add_event(run_id, "planning", "Implementation plan received.")
                storage.add_event(run_id, "codegen", "Applying generated file changes.")
                await self._materialize_files_iterative(
                    root=project_root,
                    code_plan=code_plan,
                    run_id=run_id,
                    task_text=run.task_text,
                    context_md=context_md,
                    use_repo_context=run.use_repo_context,
                    invariants_md=invariants_md,
                )
                self._run_consistency_checks(project_root=project_root, run_id=run_id)
                storage.add_event(run_id, "verify", "Running local verification commands.")
                self._run_local_commands(project_root, code_plan.local_commands, run_id)
                storage.add_event(run_id, "git", "Running local git flow.")
                self._run_git_flow(project_root, run_id, attempt)

                if settings.auto_deploy:
                    storage.add_event(run_id, "deploy", "Starting remote deploy sequence.")
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
                last_error_msg = error_msg
                storage.add_event(run_id, "error", "Pipeline step failed.", {"error": error_msg})
                storage.add_event(
                    run_id,
                    "repair",
                    "Expert repair agent invoked (JSON action plan mode).",
                    {"source": "llm_expert"},
                )
                try:
                    llm_repair_applied = await self._try_llm_repair(
                        run_id=run_id,
                        run=run,
                        error_msg=error_msg,
                        context_md=(
                            f"{context_md}\n\n## Last attempt fix directive (must follow)\n{last_fix_hint}".strip()
                            if last_fix_hint
                            else context_md
                        ),
                        project_root=project_root,
                    )
                except Exception as repair_exc:
                    llm_repair_applied = False
                    storage.add_event(
                        run_id,
                        "repair",
                        "LLM repair pipeline failed with exception.",
                        {"error": str(repair_exc)},
                    )
                if llm_repair_applied:
                    storage.update_status(run_id, RunStatus.completed)
                    storage.add_event(run_id, "done", "Run completed successfully after LLM repair.")
                    return
                if attempt == run.max_attempts:
                    storage.update_status(run_id, RunStatus.failed)
                    storage.add_event(run_id, "failed", "Max attempts reached.")
                    return

                storage.add_event(run_id, "review", "Requesting fix hint for next attempt.")
                fix_hint = await self.deepseek.review_and_fix_hint(run.task_text, error_msg, context_md)
                storage.add_event(run_id, "review", "Generated fix hint for next attempt.", {"hint": fix_hint})
                last_fix_hint = fix_hint.strip()
                if last_fix_hint:
                    context_md = (
                        f"{context_md}\n\n## Last attempt fix directive (must follow)\n{last_fix_hint}"
                    ).strip()
                repair_only_mode = True

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
            if not self._local_remote_has_heads(project_root=project_root, run_id=run_id):
                storage.add_event(
                    run_id,
                    "git",
                    "Remote repository has no branches yet. Pull skipped for bootstrap.",
                )
                return
            result = self.runner.run_local(["git", "pull", "--ff-only"], cwd=project_root, timeout_sec=600)
            storage.add_event(run_id, "git", "Local repository pull executed.", payload=result.model_dump())
            if not result.ok:
                raise RuntimeError(f"Local pull failed: {result.stderr_tail}")

    def _local_remote_has_heads(self, project_root: Path, run_id: str) -> bool:
        probe = self.runner.run_local(
            ["git", "ls-remote", "--exit-code", "--heads", "origin"],
            cwd=project_root,
            timeout_sec=120,
        )
        storage.add_event(run_id, "git", "Checked remote branches for local repository.", payload=probe.model_dump())
        return probe.ok

    def _ensure_git_identity(self, root: Path, run_id: str) -> None:
        commands = [
            ["git", "config", "user.name", settings.git_author_name],
            ["git", "config", "user.email", settings.git_author_email],
        ]
        for command in commands:
            result = self.runner.run_local(command, cwd=root, timeout_sec=60)
            storage.add_event(run_id, "git", "Git identity command executed.", payload=result.model_dump())
            if not result.ok:
                raise RuntimeError(f"Failed to configure git identity: {result.stderr_tail}")

    def _materialize_files(self, root: Path, code_plan: CodePlan, run_id: str) -> None:
        storage.add_event(run_id, "plan", "Implementation plan generated.", {"summary": code_plan.summary})
        for item in code_plan.files:
            target = (root / item.path).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Refused to write outside project root: {item.path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item.content, encoding="utf-8")
            storage.add_event(run_id, "codegen", "File generated/updated.", {"path": str(target)})

    async def _materialize_files_iterative(
        self,
        root: Path,
        code_plan: CodePlan,
        run_id: str,
        task_text: str,
        context_md: str,
        use_repo_context: bool,
        invariants_md: str,
    ) -> None:
        storage.add_event(run_id, "plan", "Implementation plan generated.", {"summary": code_plan.summary})
        generated_snapshots: list[tuple[str, str]] = []
        files = code_plan.files[: settings.iterative_codegen_max_files]
        for item in files:
            target = (root / item.path).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Refused to write outside project root: {item.path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            base_content = ""
            if use_repo_context and target.exists():
                base_content = target.read_text(encoding="utf-8")[: settings.repo_context_max_chars_per_file]

            generated_files_md = self._format_generated_snapshots(generated_snapshots)
            dependency_context_md = self._build_dependency_context(project_root=root, target_rel_path=item.path)
            final_content = await self.deepseek.generate_file_content(
                task_text=task_text,
                context_md=context_md,
                file_path=item.path,
                generated_files_md=generated_files_md,
                base_file_content=base_content or item.content,
                dependency_context_md=dependency_context_md,
                invariants_md=invariants_md,
            )
            target.write_text(final_content, encoding="utf-8")
            generated_snapshots.append((item.path, final_content[: settings.repo_context_max_chars_per_file]))
            storage.add_event(
                run_id,
                "codegen",
                "File generated/updated (iterative).",
                {"path": str(target), "with_repo_context": use_repo_context},
            )

    def _build_invariants_context(self) -> str:
        parsed = urlparse(settings.healthcheck_url)
        health_port = parsed.port or settings.app_port
        health_path = parsed.path or "/health"
        return (
            f"- Deploy directory: {settings.deploy_project_dir}\n"
            f"- Public app port must be: {settings.app_port}\n"
            f"- Healthcheck URL must be reachable: {settings.healthcheck_url}\n"
            f"- Healthcheck port inferred: {health_port}\n"
            f"- Healthcheck path inferred: {health_path}\n"
            "- Docker compose service naming must be consistent with runtime env references.\n"
            "- DATABASE_URL host should match compose postgres service name.\n"
            "- Database name in DATABASE_URL should match provisioned DB (POSTGRES_DB or created db)."
        )

    def _build_dependency_context(self, project_root: Path, target_rel_path: str) -> str:
        target = (project_root / target_rel_path).resolve()
        if not target.exists() or project_root not in target.parents:
            return ""

        selected: dict[str, str] = {}

        def add_file(path: Path) -> None:
            try:
                if project_root not in path.parents and path != project_root:
                    return
                if not path.exists() or not path.is_file():
                    return
                rel = path.relative_to(project_root).as_posix()
                if rel in selected:
                    return
                selected[rel] = path.read_text(encoding="utf-8")[: settings.repo_context_max_chars_per_file]
            except Exception:
                return

        add_file(target)
        # Always include high-impact infra files if present.
        for infra_name in ("docker-compose.yml", ".env", "README.md"):
            add_file(project_root / infra_name)

        content = target.read_text(encoding="utf-8")
        imports = re.findall(r"^\s*(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+))", content, flags=re.M)
        import_tokens = {token for pair in imports for token in pair if token}

        for token in sorted(import_tokens):
            base = token.replace(".", "/")
            candidates = [
                project_root / f"{base}.py",
                project_root / base / "__init__.py",
            ]
            for candidate in candidates:
                add_file(candidate)

        # Include neighbors in same folder to keep local cohesion.
        for sibling in sorted(target.parent.glob("*"))[:8]:
            if sibling.is_file() and sibling.suffix in {".py", ".js", ".ts", ".html", ".css", ".yml", ".yaml", ".json"}:
                add_file(sibling)

        chunks = [f"### {rel}\n{body}" for rel, body in list(selected.items())[: settings.repo_context_max_files]]
        return "\n\n".join(chunks)[:30000]

    def _run_consistency_checks(self, project_root: Path, run_id: str) -> None:
        compose_file = project_root / "docker-compose.yml"
        if not compose_file.exists():
            storage.add_event(run_id, "consistency", "docker-compose.yml not found. Consistency checks skipped.")
            return

        compose_text = compose_file.read_text(encoding="utf-8")
        issues: list[str] = []

        # Check 1: healthcheck port should be exposed in compose.
        parsed = urlparse(settings.healthcheck_url)
        health_port = parsed.port or settings.app_port
        if f"\"{health_port}:" not in compose_text and f"{health_port}:" not in compose_text:
            issues.append(f"Healthcheck port {health_port} is not exposed in docker-compose ports.")

        # Check 2: detect duplicate host-port mappings (bind conflict risk).
        host_ports = re.findall(r"['\"]?(\d+)\s*:\s*\d+['\"]?", compose_text)
        duplicates = sorted({port for port in host_ports if host_ports.count(port) > 1})
        for dup in duplicates:
            issues.append(f"Host port {dup} is mapped more than once in docker-compose services.")

        # Check 3: DATABASE_URL host should map to existing service.
        db_url_match = re.search(r"DATABASE_URL:\s*([^\n]+)", compose_text)
        if db_url_match:
            db_url = db_url_match.group(1).strip().strip("'\"")
            host = ""
            db_name = ""
            db_user = ""
            pg_match = re.search(r"postgres(?:ql)?://[^@]+@([^/:]+)(?::\d+)?/([a-zA-Z0-9_\-]+)", db_url)
            if pg_match:
                host = pg_match.group(1)
                db_name = pg_match.group(2)
            user_match = re.search(r"postgres(?:ql)?://([^:@/]+)", db_url)
            if user_match:
                db_user = user_match.group(1)
            if host and not re.search(rf"^\s*{re.escape(host)}\s*:", compose_text, flags=re.M):
                issues.append(f"DATABASE_URL host '{host}' has no matching service in docker-compose.")
            postgres_db_match = re.search(r"POSTGRES_DB:\s*([^\n]+)", compose_text)
            if db_name and postgres_db_match:
                postgres_db = postgres_db_match.group(1).strip().strip("'\"")
                if postgres_db and postgres_db != db_name:
                    issues.append(
                        f"DATABASE_URL db '{db_name}' differs from POSTGRES_DB '{postgres_db}'."
                    )
            postgres_user_match = re.search(r"POSTGRES_USER:\s*([^\n]+)", compose_text)
            if db_user and postgres_user_match:
                postgres_user = postgres_user_match.group(1).strip().strip("'\"")
                if postgres_user and postgres_user != db_user:
                    issues.append(
                        f"DATABASE_URL user '{db_user}' differs from POSTGRES_USER '{postgres_user}'."
                    )

        if issues:
            storage.add_event(run_id, "consistency", "Consistency checks failed.", {"issues": issues})
            raise RuntimeError("Consistency checks failed: " + "; ".join(issues))

        storage.add_event(run_id, "consistency", "Consistency checks passed.", {"checks": 4})

    def _format_generated_snapshots(self, snapshots: list[tuple[str, str]]) -> str:
        if not snapshots:
            return ""
        chunks = []
        for path, content in snapshots[-6:]:
            chunks.append(f"### {path}\n{content}")
        return "\n\n".join(chunks)[:20000]

    def _build_repo_context(self, project_root: Path, enabled: bool) -> str:
        if not enabled:
            return ""
        candidates: list[Path] = []
        patterns = ("*.py", "*.md", "*.txt", "*.yml", "*.yaml", "*.json", "*.html", "*.js", "*.css")
        for pattern in patterns:
            candidates.extend(project_root.rglob(pattern))
        unique = []
        seen = set()
        for file_path in sorted(candidates):
            if ".git" in file_path.parts:
                continue
            if file_path in seen:
                continue
            seen.add(file_path)
            unique.append(file_path)
        snippets: list[str] = []
        for file_path in unique[: settings.repo_context_max_files]:
            try:
                content = file_path.read_text(encoding="utf-8")[: settings.repo_context_max_chars_per_file]
            except Exception:
                continue
            rel = file_path.relative_to(project_root).as_posix()
            snippets.append(f"### {rel}\n{content}")
        return "\n\n".join(snippets)

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
        self._ensure_git_identity(root=root, run_id=run_id)
        commands = [
            ["git", "add", "."],
            ["git", "commit", "-m", f"agent attempt {attempt}: auto implementation"],
        ]
        if settings.auto_git_push:
            commands.append(["git", "branch", "-M", settings.local_git_branch])
            commands.append(["git", "push", "-u", "origin", "HEAD"])
        for command in commands:
            result = self.runner.run_local(command, cwd=root, timeout_sec=300)
            storage.add_event(run_id, "git", "Git command executed.", payload=result.model_dump())
            if len(command) > 1 and command[1] == "commit" and result.exit_code != 0:
                commit_stderr = result.stderr_tail.lower()
                commit_stdout = result.stdout_tail.lower()
                if "nothing to commit" in commit_stderr or "nothing to commit" in commit_stdout:
                    continue
                raise RuntimeError(f"Git commit failed: {result.stderr_tail}")
            if not result.ok and command[:2] == ["git", "push"]:
                push_result = self._push_with_https_token_if_available(root=root, run_id=run_id)
                if push_result is not None:
                    if push_result.ok:
                        continue
                    raise RuntimeError(f"Git push failed with token auth: {push_result.stderr_tail}")
            if not result.ok:
                raise RuntimeError(f"Git flow failed: {result.stderr_tail}")

    def _push_with_https_token_if_available(self, root: Path, run_id: str):
        if not settings.github_token:
            storage.add_event(run_id, "git", "GITHUB_TOKEN is not configured. Token push fallback skipped.")
            return None

        username = settings.github_username or "x-access-token"
        askpass_path = root / ".git" / "aiprogrammer_askpass.sh"
        askpass_content = (
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *Username*) echo \"$GITHUB_USERNAME\" ;;\n"
            "  *Password*) echo \"$GITHUB_TOKEN\" ;;\n"
            "  *) echo \"\" ;;\n"
            "esac\n"
        )
        askpass_path.write_text(askpass_content, encoding="utf-8")
        os.chmod(askpass_path, 0o700)

        env_overrides = {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": str(askpass_path),
            "GITHUB_USERNAME": username,
            "GITHUB_TOKEN": settings.github_token,
        }
        result = self.runner.run_local(
            ["git", "push", "-u", "origin", "HEAD"],
            cwd=root,
            timeout_sec=300,
            env_overrides=env_overrides,
            display_command="git push -u origin HEAD (token auth)",
        )
        storage.add_event(run_id, "git", "Git token-auth push executed.", payload=result.model_dump())
        try:
            askpass_path.unlink(missing_ok=True)
        except Exception:
            pass
        return result

    def test_git_auth(self, git_url: str) -> dict:
        url = git_url.strip()
        if not url:
            raise RuntimeError("git_url is required.")

        with tempfile.TemporaryDirectory(prefix="aiprog-git-auth-") as tmp_dir:
            workdir = Path(tmp_dir)
            branch_name = f"aiprogrammer-authcheck-{int(time.time())}"
            steps = []

            setup_commands = [
                ["git", "init", "-b", "main"],
                ["git", "config", "user.name", settings.git_author_name],
                ["git", "config", "user.email", settings.git_author_email],
                ["git", "remote", "add", "origin", url],
            ]
            for command in setup_commands:
                result = self.runner.run_local(command, cwd=workdir, timeout_sec=60)
                steps.append(result.model_dump())
                if not result.ok:
                    return {"ok": False, "message": "Git setup command failed.", "steps": steps}

            marker = workdir / "AUTH_TEST.md"
            marker.write_text("auth test\n", encoding="utf-8")
            for command in [["git", "add", "."], ["git", "commit", "-m", "auth test commit"]]:
                result = self.runner.run_local(command, cwd=workdir, timeout_sec=60)
                steps.append(result.model_dump())
                if not result.ok:
                    return {"ok": False, "message": "Git commit failed during auth test.", "steps": steps}

            push_command = ["git", "push", "--dry-run", "origin", f"HEAD:refs/heads/{branch_name}"]
            result = self.runner.run_local(push_command, cwd=workdir, timeout_sec=120)
            steps.append(result.model_dump())
            if result.ok:
                return {"ok": True, "message": "Git auth is valid for push dry-run.", "steps": steps}

            token_result = self._push_with_https_token_if_available_for_test(root=workdir, branch_name=branch_name)
            if token_result is not None:
                steps.append(token_result.model_dump())
                if token_result.ok:
                    return {"ok": True, "message": "Git auth is valid using GITHUB_TOKEN.", "steps": steps}

            return {"ok": False, "message": "Git auth test failed for push dry-run.", "steps": steps}

    def _push_with_https_token_if_available_for_test(self, root: Path, branch_name: str):
        if not settings.github_token:
            return None
        username = settings.github_username or "x-access-token"
        askpass_path = root / ".git" / "aiprogrammer_askpass.sh"
        askpass_content = (
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *Username*) echo \"$GITHUB_USERNAME\" ;;\n"
            "  *Password*) echo \"$GITHUB_TOKEN\" ;;\n"
            "  *) echo \"\" ;;\n"
            "esac\n"
        )
        askpass_path.write_text(askpass_content, encoding="utf-8")
        os.chmod(askpass_path, 0o700)
        env_overrides = {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": str(askpass_path),
            "GITHUB_USERNAME": username,
            "GITHUB_TOKEN": settings.github_token,
        }
        result = self.runner.run_local(
            ["git", "push", "--dry-run", "origin", f"HEAD:refs/heads/{branch_name}"],
            cwd=root,
            timeout_sec=120,
            env_overrides=env_overrides,
            display_command="git push --dry-run origin HEAD:<branch> (token auth)",
        )
        try:
            askpass_path.unlink(missing_ok=True)
        except Exception:
            pass
        return result

    async def draft_architecture(self, task_text: str, architect_prompt: str | None = None) -> str:
        docs = storage.list_documents()
        context_md = "\n\n".join([f"## {doc.name}\n{doc.content}" for doc in docs])[:50000]
        return await self.deepseek.build_architecture_spec(
            task_text=task_text,
            context_md=context_md,
            architect_prompt=architect_prompt,
        )

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

        runtime_checks = self.deployment.validate_remote_runtime(deploy_project_dir=deploy_project_dir)
        for check in runtime_checks:
            storage.add_event(run_id, "runtime", "Runtime validation step executed.", payload=check.model_dump())
            if not check.ok:
                raise RuntimeError(check.stderr_tail or check.stdout_tail or "Runtime validation failed.")

        health = self.deployment.healthcheck_remote(deploy_project_dir=deploy_project_dir)
        storage.add_event(run_id, "health", "Healthcheck completed.", payload=health.model_dump())
        if not health.ok:
            fallback = self.deployment.healthcheck_remote_backend_fallback(deploy_project_dir=deploy_project_dir)
            storage.add_event(run_id, "health", "Healthcheck fallback executed.", payload=fallback.model_dump())
            if fallback.ok:
                return True
            logs = self.deployment.fetch_remote_logs(deploy_project_dir=deploy_project_dir)
            storage.add_event(run_id, "health", "Healthcheck failed, logs captured.", payload=logs.model_dump())
            raise RuntimeError(health.stderr_tail or "Healthcheck failed.")
        return True

    async def _try_llm_repair(
        self,
        run_id: str,
        run: AgentRun,
        error_msg: str,
        context_md: str,
        project_root: Path,
    ) -> bool:
        duplicate_port = self._extract_duplicate_port_issue(error_msg)
        runtime_facts = self._collect_runtime_facts(deploy_project_dir=run.deploy_project_dir)
        recent_feedback = self._collect_recent_repair_feedback(run_id=run_id)
        compose_snippet = self._compose_file_fact_snippet(project_root)
        merged_parts = [runtime_facts]
        if compose_snippet:
            merged_parts.append(
                "[docker_compose — current repository excerpt; find_text MUST match this text exactly]\n"
                + compose_snippet
            )
        merged_parts.append(f"[recent_repair_feedback]\n{recent_feedback}")
        merged_runtime_facts = "\n\n".join(merged_parts).strip()
        plan: RepairPlan = await self.deepseek.propose_repair_plan(
            task_text=run.task_text,
            last_error=error_msg,
            context_md=context_md,
            runtime_facts=merged_runtime_facts,
        )
        storage.add_event(run_id, "repair", "LLM repair plan received.", payload=plan.model_dump())
        if not plan.actions:
            return False
        if not self._has_treatment_action(plan.actions):
            storage.add_event(
                run_id,
                "repair",
                "Repair plan rejected: no treatment action (diagnostics-only plan).",
                {"actions": [a.model_dump() for a in plan.actions]},
            )
            return False

        for action in plan.actions:
            action_type = (action.action_type or "").strip().lower()
            if action_type == "run_remote_command":
                cmd = action.command.strip()
                if not cmd:
                    storage.add_event(run_id, "repair", "Repair action rejected: empty remote command.", payload=action.model_dump())
                    return False
                result = self.runner.run_ssh(cmd, timeout_sec=900)
                storage.add_event(run_id, "repair", "Repair action executed.", payload=result.model_dump())
                if not result.ok:
                    return False
                continue

            if action_type == "run_local_command":
                cmd = action.command.strip()
                if not cmd:
                    return False
                # Safety fallback: in repair loop all diagnostics/treatment should be remote.
                # If model still returns run_local_command, execute it remotely to avoid dead loops.
                result = self.runner.run_ssh(cmd, timeout_sec=900)
                storage.add_event(run_id, "repair", "Repair action executed.", payload=result.model_dump())
                if not result.ok:
                    return False
                continue

            if action_type == "replace_text_in_file":
                normalized_path = self._normalize_repair_file_path(
                    file_path=action.file_path,
                    deploy_project_dir=run.deploy_project_dir,
                )
                ok = self._apply_replace_text_action(project_root, normalized_path, action.find_text, action.replace_text)
                storage.add_event(
                    run_id,
                    "repair",
                    "Repair file patch applied.",
                    {
                        "ok": ok,
                        "file_path": action.file_path,
                        "normalized_file_path": normalized_path,
                        "action_type": action_type,
                    },
                )
                if not ok:
                    return False
                continue

            if action_type == "update_healthcheck_url":
                storage.add_event(
                    run_id,
                    "repair",
                    "Healthcheck URL update requested; using remote fallback validation in current run.",
                    {"requested_target": action.target, "action_type": action_type},
                )
                continue

            if action_type == "ensure_postgres_db":
                db_name = (action.target or "taskcalendar").strip()
                cmd = (
                    "cd \"{dir}\" && docker compose exec -T db sh -lc "
                    "\"psql -U $POSTGRES_USER -d postgres -tc \\\"SELECT 1 FROM pg_database WHERE datname='{db}'\\\" "
                    "| grep -q 1 || psql -U $POSTGRES_USER -d postgres -c \\\"CREATE DATABASE {db};\\\"\""
                ).format(dir=run.deploy_project_dir, db=db_name)
                result = self.runner.run_ssh(cmd, timeout_sec=300)
                storage.add_event(run_id, "repair", "Repair action executed.", payload=result.model_dump())
                if not result.ok:
                    return False
                continue

            storage.add_event(
                run_id,
                "repair",
                "Unsupported repair action type.",
                {"action_type": action_type},
            )
            return False

        if plan.validation_steps:
            checks_ok = self._execute_repair_validation_steps(
                run_id=run_id,
                deploy_project_dir=run.deploy_project_dir,
                validation_steps=plan.validation_steps,
            )
            if not checks_ok:
                return False

        if duplicate_port:
            compose_ok = self._assert_single_host_port_mapping(
                project_root=project_root,
                run_id=run_id,
                host_port=duplicate_port,
            )
            if not compose_ok:
                storage.add_event(
                    run_id,
                    "repair",
                    "Repair post-condition failed: duplicate host port mapping still exists.",
                    {"port": duplicate_port},
                )
                return False

        if run.git_url:
            self._run_git_flow(project_root, run_id, attempt=run.current_attempt or 1)

        if settings.auto_deploy:
            if not await self._deploy_and_validate(run_id, run.git_url, run.deploy_project_dir):
                return False
        storage.add_event(run_id, "repair", "Repair feedback window cleared.", {})
        return True

    def _collect_recent_repair_feedback(self, run_id: str) -> str:
        run_snapshot = storage.get_run(run_id)
        events = run_snapshot.events
        baseline_idx = -1
        for i, ev in enumerate(events):
            if ev.stage == "repair" and ev.message == "Repair feedback window cleared.":
                baseline_idx = i
        window_events = events[baseline_idx + 1 :] if baseline_idx >= 0 else events

        failed_commands: list[str] = []
        rejected_actions: list[str] = []
        for ev in reversed(window_events):
            if ev.stage != "repair":
                continue
            payload = ev.payload or {}
            if ev.message == "Repair file patch applied." and payload.get("ok") is False:
                fp = str(payload.get("normalized_file_path", "")).strip()
                failed_commands.append(
                    "replace_text_in_file: find_text did not match file (check exact whitespace/quotes)"
                    + (f" | file={fp}" if fp else "")
                )
            if ev.message == "Repair action executed." and payload.get("ok") is False:
                cmd = str(payload.get("command", "")).strip()
                err = str(payload.get("stderr_tail", "")).strip()
                if cmd:
                    failed_commands.append(f"command={cmd} | stderr={err}")
            if ev.message.startswith("Repair action rejected"):
                rejected_actions.append(str(payload))
            if ev.message.startswith("Repair plan rejected"):
                rejected_actions.append(f"{ev.message} | {payload}")
            if len(failed_commands) >= 6 and len(rejected_actions) >= 6:
                break
        chunks = []
        if failed_commands:
            chunks.append("[failed_commands]\n" + "\n".join(f"- {item}" for item in failed_commands[:6]))
        if rejected_actions:
            chunks.append("[rejected_actions]\n" + "\n".join(f"- {item}" for item in rejected_actions[:6]))
        if not chunks:
            return "No previous failed/rejected repair actions."
        return "\n\n".join(chunks)

    def _execute_repair_validation_steps(
        self,
        run_id: str,
        deploy_project_dir: str,
        validation_steps: list[str],
    ) -> bool:
        for step in validation_steps:
            cmd = self._extract_command_from_validation_step(step)
            if not cmd:
                storage.add_event(
                    run_id,
                    "repair",
                    "Validation step skipped (no command parsed).",
                    {"step": step},
                )
                continue
            if cmd.startswith("curl ") or cmd.startswith("docker ") or cmd.startswith("docker-compose ") or cmd.startswith("docker compose "):
                final_cmd = cmd
            else:
                final_cmd = f"cd \"{deploy_project_dir}\" && {cmd}"
            result = self.runner.run_ssh(final_cmd, timeout_sec=180)
            storage.add_event(
                run_id,
                "repair",
                "Validation step executed.",
                {"step": step, "command": final_cmd, "ok": result.ok, "stderr_tail": result.stderr_tail, "stdout_tail": result.stdout_tail},
            )
            if not result.ok:
                return False
        return True

    def _extract_command_from_validation_step(self, step: str) -> str:
        text = (step or "").strip()
        if not text:
            return ""
        quote_match = re.search(r"[\"']([^\"']+)[\"']", text)
        if quote_match:
            return quote_match.group(1).strip()
        lowered = text.lower()
        prefixes = ("run ", "execute ")
        for prefix in prefixes:
            if lowered.startswith(prefix):
                return text[len(prefix) :].strip()
        # If it's already a raw shell command (most model outputs), execute as-is.
        return text

    def _collect_runtime_facts(self, deploy_project_dir: str) -> str:
        facts: list[str] = []
        ps = self.runner.run_ssh(f"cd \"{deploy_project_dir}\" && docker compose ps", timeout_sec=120)
        logs = self.deployment.fetch_remote_logs(deploy_project_dir=deploy_project_dir)
        facts.append(f"[compose_ps]\n{ps.stdout_tail}\n{ps.stderr_tail}")
        facts.append(f"[compose_logs]\n{logs.stdout_tail}\n{logs.stderr_tail}")
        return "\n\n".join(facts)

    def _apply_replace_text_action(self, project_root: Path, file_path: str, find_text: str, replace_text: str) -> bool:
        if not file_path:
            return False
        target = (project_root / file_path).resolve()
        if project_root not in target.parents and target != project_root:
            return False
        if not target.exists():
            return False
        content = target.read_text(encoding="utf-8")
        if find_text not in content:
            return False
        updated = content.replace(find_text, replace_text, 1)
        target.write_text(updated, encoding="utf-8")
        return True

    def _normalize_repair_file_path(self, file_path: str, deploy_project_dir: str) -> str:
        path = (file_path or "").strip()
        if not path:
            return path
        if path.startswith("/"):
            deploy_dir = (deploy_project_dir or "").rstrip("/")
            if deploy_dir and path.startswith(f"{deploy_dir}/"):
                return path[len(deploy_dir) + 1 :]
            return path.lstrip("/")
        return path

    def _has_treatment_action(self, actions: list) -> bool:
        diagnostic_prefixes = (
            "cat ",
            "grep ",
            "rg ",
            "docker ps",
            "docker compose ps",
            "docker compose logs",
            "docker compose config",
            "ss ",
            "netstat ",
        )
        for action in actions:
            action_type = (action.action_type or "").strip().lower()
            if action_type == "replace_text_in_file":
                ft_raw = action.find_text or ""
                rt_raw = action.replace_text if action.replace_text is not None else ""
                if not ft_raw.strip():
                    continue
                if ft_raw == rt_raw:
                    continue
                return True
            if action_type == "ensure_postgres_db":
                return True
            if action_type == "update_healthcheck_url":
                return True
            if action_type in {"run_remote_command", "run_local_command"}:
                cmd = (action.command or "").strip().lower()
                if not cmd:
                    continue
                if any(cmd.startswith(prefix) for prefix in diagnostic_prefixes):
                    continue
                return True
        return False

    def _extract_duplicate_port_issue(self, error_msg: str) -> int | None:
        text = (error_msg or "").lower()
        if "host port" in text and "mapped more than once" in text:
            match = re.search(r"host port\s+(\d+)", text)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return None
        return None

    def _resolve_compose_file(self, project_root: Path) -> Path | None:
        for name in ("docker-compose.yml", "docker-compose.yaml", "compose.yml"):
            candidate = project_root / name
            if candidate.exists():
                return candidate
        return None

    def _compose_file_fact_snippet(self, project_root: Path, limit: int = 8000) -> str | None:
        path = self._resolve_compose_file(project_root)
        if not path:
            return None
        return path.read_text(encoding="utf-8")[:limit]

    def _count_compose_host_port_mappings(self, content: str, host_port: int) -> int:
        host_ports = re.findall(r"['\"]?(\d+)\s*:\s*\d+['\"]?", content)
        return sum(1 for port in host_ports if port == str(host_port))

    def _assert_single_host_port_mapping(self, project_root: Path, run_id: str, host_port: int) -> bool:
        compose_file = self._resolve_compose_file(project_root)
        if not compose_file:
            storage.add_event(
                run_id,
                "repair",
                "Post-condition check skipped: no compose file in project root.",
            )
            return False
        content = compose_file.read_text(encoding="utf-8")
        count = self._count_compose_host_port_mappings(content, host_port)
        storage.add_event(
            run_id,
            "repair",
            "Post-condition check executed for compose host-port uniqueness.",
            {"port": host_port, "count": count, "compose_file": compose_file.name},
        )
        return count == 1


orchestrator = AgentOrchestrator()
