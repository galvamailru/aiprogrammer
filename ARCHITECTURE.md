# Agentic Architecture

## High-level flow

```mermaid
flowchart TD
    UI[Web UI] --> API[FastAPI]
    API --> STORE[InMemory Storage]
    API --> ORCH[Agent Orchestrator]
    ORCH --> DS[DeepSeek Client]
    ORCH --> TERM[Terminal Runner]
    ORCH --> DEPLOY[Deployment Service]
    TERM --> LOCAL[Local Project Directory]
    DEPLOY --> SSH[Linux Server via SSH]
    DEPLOY --> HEALTH[HTTP Healthcheck]
```

## Retry loop

```mermaid
stateDiagram-v2
    [*] --> Intake
    Intake --> Attempt
    Attempt --> Plan
    Plan --> Codegen
    Codegen --> Verify
    Verify --> GitFlow
    GitFlow --> Deploy
    Deploy --> Healthcheck
    Healthcheck --> Done : success
    Healthcheck --> Review : failed
    Review --> Attempt : attempts left
    Review --> Failed : no attempts left
    Done --> [*]
    Failed --> [*]
```

## Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant W as Web UI
    participant A as FastAPI
    participant O as Orchestrator
    participant D as DeepSeek
    participant R as Runner
    participant S as Linux Server

    U->>W: Enter business task
    W->>A: POST /api/runs
    A->>O: start_run(task)
    O->>D: build_code_plan(task, markdown context)
    D-->>O: files + summary + commands
    O->>R: local git commands
    O->>S: ssh git pull
    O->>S: ssh docker compose up -d --build
    O->>A: save events/state
    A-->>W: run status/events
```

## Environment variables

- `DEEPSEEK_API_KEY`: API key for planner/reviewer calls
- `LINUX_HOST`, `LINUX_USERNAME`, `LINUX_PASSWORD`: remote deploy access
- `DEPLOY_PROJECT_DIR`: server directory where docker compose runs
- `LOCAL_PROJECT_DIR`: local project where generated files are materialized
- `AUTO_GIT_PUSH`: optional automatic push after commit
- `AUTO_DEPLOY`: toggle remote deploy stage
- `MAX_FIX_ATTEMPTS`: retry loop length
