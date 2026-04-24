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
    healthcheck_url: str = "http://127.0.0.1:8000/health"
    healthcheck_timeout_seconds: int = 120

    local_project_dir: str = "./target_project"
    local_git_branch: str = "main"
    auto_git_push: bool = False
    auto_deploy: bool = True
    max_fix_attempts: int = 3
    git_author_name: str = "AI Programmer Bot"
    git_author_email: str = "bot@example.com"

    app_host: str = "0.0.0.0"
    app_port: int = 8080

    upload_dir: str = "./uploads"
    run_data_dir: str = "./runs"

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
