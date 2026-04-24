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
2. Set git repository URL and run `Test Git auth`
3. Set deploy directory on Linux server
4. Enter business task and start run

## Core architecture

- `app/orchestrator.py` - state machine and retry loop
- `app/deepseek_client.py` - model calls
- `app/terminal_runner.py` - local + SSH command execution
- `app/deployment.py` - deploy and health flow
- `app/storage.py` - run metadata and context docs

## Notes

- This is an MVP scaffold intended for controlled environments.
- For production, add:
  - secrets manager integration
  - RBAC and policy engine
  - CI gates and branch protections
  - full observability stack
