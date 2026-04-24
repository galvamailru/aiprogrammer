from __future__ import annotations

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

    def remediation_rules(self) -> list[dict]:
        return [
            {
                "id": "port_allocated",
                "signatures": ["port is already allocated", "bind for 0.0.0.0"],
                "action": "docker compose down --remove-orphans && docker compose up -d --build",
            },
            {
                "id": "pg_data_version_mismatch",
                "signatures": ["database files are incompatible with server", "initialized by postgresql version"],
                "action": "docker compose down --remove-orphans -v && docker compose up -d --build",
            },
            {
                "id": "connection_refused_health",
                "signatures": ["[errno 111] connection refused", "connection refused"],
                "action": "inspect compose ports and application listen address per architecture spec",
            },
        ]

    def detect_remediation_rule(self, error_text: str) -> dict | None:
        lowered = (error_text or "").lower()
        for rule in self.remediation_rules():
            if any(signature in lowered for signature in rule["signatures"]):
                return rule
        return None

    def run_remediation(self, rule_id: str, deploy_project_dir: str) -> list[CommandResult]:
        safe_dir = deploy_project_dir.replace('"', "")
        if rule_id == "port_allocated":
            return [
                self.runner.run_ssh(f"cd \"{safe_dir}\" && docker compose down --remove-orphans", timeout_sec=600),
                self.runner.run_ssh(f"cd \"{safe_dir}\" && docker compose up -d --build", timeout_sec=1800),
            ]
        if rule_id == "pg_data_version_mismatch":
            return [
                self.runner.run_ssh(f"cd \"{safe_dir}\" && docker compose down --remove-orphans -v", timeout_sec=600),
                self.runner.run_ssh(f"cd \"{safe_dir}\" && docker compose up -d --build", timeout_sec=1800),
            ]
        if rule_id == "connection_refused_health":
            return [
                self.runner.run_ssh(f"cd \"{safe_dir}\" && docker compose ps", timeout_sec=120),
            ]
        return []

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
