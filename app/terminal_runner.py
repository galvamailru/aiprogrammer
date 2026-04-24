from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import paramiko

from .config import settings
from .models import CommandResult


def _tail(text: str, limit: int = 12000) -> str:
    return text[-limit:]


class TerminalRunner:
    def run_local(
        self,
        command: list[str],
        cwd: Path,
        timeout_sec: int = 300,
        env_overrides: dict[str, str] | None = None,
        display_command: str | None = None,
    ) -> CommandResult:
        t0 = time.time()
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            shell=False,
            check=False,
        )
        elapsed = round(time.time() - t0, 3)
        return CommandResult(
            ok=proc.returncode == 0,
            command=display_command or " ".join(command),
            exit_code=proc.returncode,
            stdout_tail=_tail(proc.stdout),
            stderr_tail=_tail(proc.stderr),
            duration_sec=elapsed,
            source="local",
        )

    def run_ssh(self, command: str, timeout_sec: int = 300) -> CommandResult:
        t0 = time.time()
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(
                hostname=settings.linux_host,
                port=settings.linux_port,
                username=settings.linux_username,
                password=settings.linux_password,
                timeout=20,
            )
            stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout_sec)
            out_text = stdout.read().decode("utf-8", errors="ignore")
            err_text = stderr.read().decode("utf-8", errors="ignore")
            code = stdout.channel.recv_exit_status()
        finally:
            ssh.close()

        elapsed = round(time.time() - t0, 3)
        return CommandResult(
            ok=code == 0,
            command=command,
            exit_code=code,
            stdout_tail=_tail(out_text),
            stderr_tail=_tail(err_text),
            duration_sec=elapsed,
            source="ssh",
        )
