from __future__ import annotations

import time

import httpx

from .config import settings
from .models import CommandResult
from .terminal_runner import TerminalRunner


class DeploymentService:
    def __init__(self, runner: TerminalRunner) -> None:
        self.runner = runner

    def deploy_remote(self, git_url: str, deploy_project_dir: str) -> list[CommandResult]:
        safe_dir = deploy_project_dir.replace('"', "")
        safe_repo = git_url.replace('"', "")
        commands = [
            (
                f"if [ ! -d \"{safe_dir}/.git\" ]; then "
                f"mkdir -p \"{safe_dir}\" && git clone \"{safe_repo}\" \"{safe_dir}\"; "
                f"else cd \"{safe_dir}\" && "
                f"if git ls-remote --exit-code --heads origin >/dev/null 2>&1; then "
                f"git fetch --all && git pull --ff-only; "
                f"else echo 'Remote has no branches yet, pull skipped.'; fi; fi"
            ),
            f"cd \"{safe_dir}\" && docker compose up -d --build",
            f"cd \"{safe_dir}\" && docker compose ps",
        ]
        results: list[CommandResult] = []
        for command in commands:
            result = self.runner.run_ssh(command, timeout_sec=1200)
            results.append(result)
            if not result.ok:
                break
        return results

    def fetch_remote_logs(self, deploy_project_dir: str) -> CommandResult:
        safe_dir = deploy_project_dir.replace('"', "")
        command = f"cd \"{safe_dir}\" && docker compose logs --tail=200"
        return self.runner.run_ssh(command, timeout_sec=300)

    def validate_remote_runtime(self, deploy_project_dir: str) -> list[CommandResult]:
        safe_dir = deploy_project_dir.replace('"', "")
        checks: list[CommandResult] = []

        ps_result = self.runner.run_ssh(f"cd \"{safe_dir}\" && docker compose ps", timeout_sec=120)
        checks.append(ps_result)
        if not ps_result.ok:
            return checks

        ps_text = f"{ps_result.stdout_tail}\n{ps_result.stderr_tail}".lower()
        bad_states = [" exited ", " restarting ", " unhealthy "]
        if any(state in ps_text for state in bad_states):
            checks.append(
                CommandResult(
                    ok=False,
                    command="runtime state validation",
                    exit_code=1,
                    stdout_tail=ps_result.stdout_tail,
                    stderr_tail="Compose has unhealthy/exited/restarting containers.",
                    duration_sec=0.0,
                    source="analyzer",
                )
            )
            return checks

        logs_result = self.fetch_remote_logs(deploy_project_dir=deploy_project_dir)
        checks.append(logs_result)
        if not logs_result.ok:
            return checks

        log_text = f"{logs_result.stdout_tail}\n{logs_result.stderr_tail}".lower()
        error_signatures = [
            "traceback",
            "operationalerror",
            "connection refused",
            "name or service not known",
            "could not translate host name",
        ]
        if any(signature in log_text for signature in error_signatures):
            checks.append(
                CommandResult(
                    ok=False,
                    command="runtime log validation",
                    exit_code=1,
                    stdout_tail=logs_result.stdout_tail,
                    stderr_tail="Critical error signature detected in container logs.",
                    duration_sec=0.0,
                    source="analyzer",
                )
            )
        else:
            checks.append(
                CommandResult(
                    ok=True,
                    command="runtime log validation",
                    exit_code=0,
                    stdout_tail="No critical runtime signatures detected.",
                    stderr_tail="",
                    duration_sec=0.0,
                    source="analyzer",
                )
            )

        return checks

    def healthcheck(self) -> CommandResult:
        deadline = time.time() + settings.healthcheck_timeout_seconds
        last_error = ""
        while time.time() < deadline:
            try:
                response = httpx.get(settings.healthcheck_url, timeout=10)
                if 200 <= response.status_code < 300:
                    return CommandResult(
                        ok=True,
                        command=f"GET {settings.healthcheck_url}",
                        exit_code=0,
                        stdout_tail=response.text[-4000:],
                        stderr_tail="",
                        duration_sec=0.0,
                        source="http",
                    )
                last_error = f"HTTP {response.status_code}: {response.text[:400]}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(5)
        return CommandResult(
            ok=False,
            command=f"GET {settings.healthcheck_url}",
            exit_code=1,
            stdout_tail="",
            stderr_tail=last_error,
            duration_sec=float(settings.healthcheck_timeout_seconds),
            source="http",
        )
