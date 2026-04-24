from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    linux_host: str = "127.0.0.1"
    linux_port: int = 22
    linux_username: str = "deploy"
    linux_password: str = ""

    deploy_project_dir: str = "/opt/myapp"
    deploy_verify_max_commands: int = 15
    deploy_verify_ssh_timeout_sec: int = 300

    local_project_dir: str = "./target_project"
    local_git_branch: str = "main"
    auto_git_push: bool = False
    auto_deploy: bool = True
    max_fix_attempts: int = 3
    git_author_name: str = "AI Programmer Bot"
    git_author_email: str = "bot@example.com"
    github_username: str = ""
    github_token: str = ""
    architect_system_prompt: str = (
        "You are a senior solution architect. "
        "Return strict JSON with key: architecture_spec_markdown. "
        "The markdown must include sections: Goal, Tech stack, Functional requirements, "
        "Non-functional requirements, Project structure requirements, Run requirements, Acceptance criteria, "
        "and a dedicated section exactly titled: ## Deploy verification (final stage). "
        "That section is mandatory: it describes how the orchestrator will prove the system works on the Linux server "
        "after `docker compose up` succeeds and container logs show no critical errors. "
        "Include: (1) Preconditions (e.g. which service exposes which host port, base URL on 127.0.0.1). "
        "(2) Ordered verification steps as concrete shell commands runnable on the deployment host inside the project directory "
        "(use `docker compose ...`, `curl -fsS`, `psql`, etc. as appropriate to the stack). "
        "(3) Expected output or HTTP status for each step and what constitutes failure. "
        "(4) A short smoke path for the main user-facing feature (e.g. one API call or one page fetch) if applicable. "
        "Do not assume a fixed /health URL unless you specify it for this project. "
        "Commands must be safe (read-only checks, no destructive ops)."
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8080

    upload_dir: str = "./uploads"
    run_data_dir: str = "./runs"
    repo_context_max_files: int = 12
    repo_context_max_chars_per_file: int = 5000
    iterative_codegen_max_files: int = 30
    fix_only_max_plan_files: int = 12

    fix_only_plan_system_prompt: str = (
        "You are a senior engineer applying a surgical patch to an existing codebase. "
        "Return strict JSON with keys: summary, files, local_commands. "
        "files is an array of objects {path, content}: for every path that MUST change, give full new file content "
        "(a later pass may refine wording, but stay close to minimal edits). "
        "Include ONLY files that the change request requires—never list untouched files as placeholders. "
        "Do not rename public HTTP routes, env vars, or docker-compose service names unless the change request explicitly says so. "
        "Preserve response/request shapes and backward compatibility unless the request mandates a breaking change. "
        "If the repository context is thin, still choose the smallest set of paths that plausibly implement the fix. "
        "local_commands may be an empty array. "
        "The user message states the maximum allowed number of files in files—obey it."
    )

    @property
    def local_project_path(self) -> Path:
        return Path(self.local_project_dir).resolve()

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir).resolve()

    @property
    def run_data_path(self) -> Path:
        return Path(self.run_data_dir).resolve()


settings = Settings()
