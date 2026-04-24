# AI Programmer

FastAPI application that implements an agentic pipeline:
- business task intake
- markdown context upload (API specs, technical docs, requirements)
- planning and implementation draft via DeepSeek
- local code materialization into project directory
- optional git push
- remote deploy via SSH + docker compose
- health/log analysis and automatic retry loop
- git repository URL passed from UI per run
- deploy directory passed from UI per run

## Quick start

1. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure variables:

```bash
cp .env.example .env
```

Fill:
- DeepSeek API credentials
- Linux server credentials
- local project directory
- deploy directory on server
- git author identity for automated commits
- GitHub credentials for HTTPS push from container (`GITHUB_USERNAME`, `GITHUB_TOKEN`)

3. Run app:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

4. Open UI:

`http://localhost:8080`

## Run in Docker

1. Copy env file and fill credentials:

```bash
cp .env.example .env
```

2. Build and run:

```bash
docker compose up -d --build
```

3. Open:

`http://localhost:8080`

4. Watch runtime logs:

```bash
docker logs -f aiprogrammer
```

## UI flow

1. Upload markdown context files (`.md`)
2. Set or edit architect agent prompt
3. Enter business task and click `Generate TZ`
4. Review/edit generated architecture TZ and confirm (`Согласовано`)
5. Set git repository URL and run `Test Git auth`
6. Set deploy directory on Linux server
7. Start run after architecture approval

## Core architecture

- `app/orchestrator.py` - state machine and retry loop
- `app/deepseek_client.py` - model calls
- `app/terminal_runner.py` - local + SSH command execution
- `app/deployment.py` - deploy and health flow
- `app/storage.py` - run metadata and context docs

## Notes

- This is an MVP scaffold intended for controlled environments.
- Auto-remediation rules are implemented for common deploy failures:
  - `port is already allocated` -> compose down/remove-orphans + rebuild/up
  - `database files are incompatible with server` -> compose down with volumes + rebuild/up
  - `connection refused` -> remote fallback healthcheck by detected backend mapped port
- LLM repair contract (`RepairAction`) is implemented for non-matching failures:
  - `run_remote_command`
  - `run_local_command`
  - `replace_text_in_file`
  - `update_healthcheck_url`
- For production, add:
  - secrets manager integration
  - RBAC and policy engine
  - CI gates and branch protections
  - full observability stack
