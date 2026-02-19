# Velox Optimization Engine

Python-first code optimization platform that combines:
- static code scanning,
- LLM-based semantic/performance analysis,
- RL policy-based optimization strategy selection,
- optional validation and ranking,
- offline DQN pretraining with dataset generation and evaluation gates.

The system currently focuses on **suggestions** (not auto-applying patches).

## Feature Map

| Capability | What it does | Primary modules |
|---|---|---|
| Inline/file optimization API | Accepts Python code or file path and returns ranked optimization suggestions | `code/code_optimizer_ai/api_main.py`, `code/code_optimizer_ai/ml/rl_optimizer.py` |
| Repository scan (`/scan/repository`) | Clones a GitHub repo and returns scan metrics, complexity hotspots, and optimization candidates | `code/code_optimizer_ai/api_main.py`, `code/code_optimizer_ai/core/code_scanner.py` |
| Repository optimize (`/optimize/repository`) | Clones a GitHub repo, selects top Python files, and returns per-file optimization suggestions | `code/code_optimizer_ai/api_main.py`, `code/code_optimizer_ai/ml/rl_optimizer.py` |
| Cross-file context | Adds import graph + related file snippets for repository optimization | `code/code_optimizer_ai/api_main.py` |
| Batch coordinated suggestions | For connected repository files, requests coordinated multi-file suggestions in one LLM pass | `code/code_optimizer_ai/api_main.py`, `code/code_optimizer_ai/core/llm_analyzer.py` |
| LLM routing/fallback | Routes analysis/generation through OpenRouter models, then Ollama fallback | `code/code_optimizer_ai/core/llm_gateway.py` |
| RL decisioning | Chooses optimization action type with DQN policy + action confidence | `code/code_optimizer_ai/ml/rl_optimizer.py`, `code/code_optimizer_ai/ml/policy_manager.py`, `code/code_optimizer_ai/ml/rl_agent.py` |
| Validation engine | Syntax gate, optional pytest gate in sandbox copy, micro-benchmark for runtime/memory deltas | `code/code_optimizer_ai/core/validation_engine.py` |
| Multi-agent orchestrator | Monitor/Analyzer/Optimizer/Learning agents with queues and periodic loop | `code/code_optimizer_ai/agents/pipeline_orchestrator.py` |
| Performance profiling | cProfile + memory/cpu metrics + baseline comparison | `code/code_optimizer_ai/core/performance_profiler.py` |
| Offline RL training pipeline | Generate transitions, split train/holdout, pretrain, gate, publish/reject model | `code/code_optimizer_ai/generate_training_data.py`, `code/code_optimizer_ai/pretrain_dqn.py`, `code/code_optimizer_ai/ml/training_runner.py` |
| Storage + cache | Postgres records/episodes/patterns and Redis caching/experience buffers with key prefixing | `code/code_optimizer_ai/database/connection.py`, `code/code_optimizer_ai/database/migrate.py` |

## High-Level Architecture

```text
Client
  |
  v
FastAPI (api_main.py)
  |-- Auth/rate-limit/path guards
  |-- /optimize, /scan, /train, /metrics, /pipeline/*
  |
  +--> CodeScanner (AST + complexity + imports)
  +--> CodeAnalyzerLLM (LLM prompts + JSON parsing)
  |      |
  |      +--> LLMGateway (OpenRouter primary/secondary -> Ollama fallback)
  |
  +--> RLCodeOptimizer
  |      |
  |      +--> PolicyManager (active + shadow DQN)
  |      +--> DQNAgent (action recommendation)
  |      +--> ValidationEngine (optional ranking signal)
  |
  +--> PipelineOrchestrator (Monitor/Analyzer/Optimizer/Learning agents)
  |
  +--> DatabaseManager (Postgres) + CacheManager (Redis)
```

## Request Flows

### 1. Inline/file optimization (`POST /optimize`, `POST /optimize/file`)
1. API validates payload size, auth, and optional unit test command override policy.
2. `RLCodeOptimizer.optimize_code(...)` runs:
   - LLM analysis (`analyze_code`)
   - state-vector build (27-dim production semantics)
   - DQN action recommendation
   - heuristic transform or LLM suggestion generation
   - optional validation/ranking
3. Returns serialized ranked suggestions.

### 2. Repository optimization (`POST /optimize/repository`)
1. Clone GitHub repository (HTTPS only, validated owner/repo/ref).
2. Scan files and select top targets by performance candidates + complexity ranking.
3. Build project context (module/import graph).
4. Optional:
   - `enable_cross_file_context`: per-file related module snippets.
   - `enable_batch_coordination`: connected-file grouped LLM suggestions.
5. Fallback per file to RL optimizer when coordinated batch output is missing.
6. Return per-file suggestion results.

## API Surface

Protected endpoints require `X-API-Token` when `API_AUTH_TOKEN` is configured.

### Public
- `GET /`
- `GET /health`
- `GET /suggestions/categories`

### Protected
- `GET /health/details`
- `POST /optimize`
- `POST /optimize/file`
- `POST /scan`
- `POST /scan/repository`
- `POST /optimize/repository`
- `GET /pipeline/status`
- `POST /train`
- `GET /train/{job_id}`
- `GET /train`
- `GET /metrics`
- `POST /monitor/directory`
- `GET /database/status`
- `POST /database/migrate`
- `POST /database/seed` (debug-only behavior in handler)
- `GET /demo/database`

## Project Layout

```text
code/code_optimizer_ai/
  api_main.py                # FastAPI entrypoint + endpoint orchestration
  agents/
    pipeline_orchestrator.py # Multi-agent async orchestration
  core/
    code_scanner.py          # AST scan + file complexity/import stats
    llm_gateway.py           # Provider routing/fallback
    llm_analyzer.py          # Analysis + suggestion prompts/parsing
    performance_profiler.py  # Runtime/memory/CPU instrumentation
    validation_engine.py     # Syntax/tests/benchmark validation pipeline
  ml/
    rl_optimizer.py          # Main online suggestion engine
    rl_agent.py              # DQN implementation + replay
    policy_manager.py        # Active/shadow policy management
    rl_environment.py        # Experimental online env
    training_runner.py       # Primary offline training pipeline
    training_semantics.py    # Canonical action set + state/reward semantics
  database/
    connection.py            # Postgres/Redis managers
    migrate.py               # Schema + seeding helpers
  generate_training_data.py  # Legacy episode -> transition JSONL converter
  pretrain_dqn.py            # Offline pretraining CLI from JSONL transitions
  run.py                     # Python runtime entry
  run.sh                     # Shell runtime entry
```

## Setup (Local)

### Prerequisites
- Python 3.12+
- PostgreSQL
- Redis
- Optional local LLM: Ollama

### 1) Create environment and install dependencies
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r code/code_optimizer_ai/requirements.txt
```

### 2) Configure environment file
Settings load from:
- `code/code_optimizer_ai/.env`

Use:
- `code/code_optimizer_ai/.env.example` as template.

Important required values:
- `APP_NAME`
- `APP_VERSION`
- `SECRET_KEY`
- `DATABASE_URL`
- `REDIS_URL`

Recommended security values:
- `REQUIRE_AUTH_TOKEN=True`
- `API_AUTH_TOKEN=<strong token>`
- `ENABLE_API_DOCS=False`

### 3) Run API
```powershell
python code/code_optimizer_ai/run.py
```

or

```bash
./code/code_optimizer_ai/run.sh
```

## Training and Model Lifecycle

Primary training path is **offline**.

### CLI path
1. Generate transitions:
```bash
python code/code_optimizer_ai/generate_training_data.py \
  --input-dir data/training \
  --output-jsonl data/training/pretrain_transitions.jsonl \
  --synthetic-samples 10000 \
  --seed 42
```

2. Pretrain DQN:
```bash
python code/code_optimizer_ai/pretrain_dqn.py \
  --input data/training/pretrain_transitions.jsonl \
  --output models/rl_policy/dqn_model.pth \
  --epochs 40 \
  --steps-per-epoch 1000 \
  --batch-size 64 \
  --shuffle \
  --fresh-start
```

### API path
- `POST /train` starts job (background or sync).
- `GET /train/{job_id}` polls status.
- `GET /train` lists recent jobs.

Training gate behavior:
- Evaluates candidate model on holdout TD error.
- Compares against current deployed checkpoint.
- Publishes only if gate passes; otherwise rejects candidate.

See `docs/training_runbook.md` for operational details.

## Testing

Run integration tests:
```bash
pytest code/code_optimizer_ai/tests/test_integration.py -q -p no:cacheprovider
```

Run selected RL/data tests:
```bash
pytest code/code_optimizer_ai/tests/test_pretrain_dqn.py \
       code/code_optimizer_ai/tests/test_generate_training_data.py \
       code/code_optimizer_ai/tests/test_cache_key_prefix.py -q -p no:cacheprovider
```

## Security and Operational Notes

- Auth uses token header `X-API-Token` when enabled.
- Path access is constrained by `ALLOWED_CODE_ROOTS`.
- Repository cloning is restricted to HTTPS GitHub URLs.
- API has in-process request rate limiting.
- Redis keys are namespaced using `REDIS_KEY_PREFIX` to reduce cross-app collisions.
- Validation uses a sandbox-copy strategy and restricted command parsing; it is safer than direct execution but still not equivalent to a hardened container sandbox.

## Current Scope and Limits

- Python files only (`.py`) for scanning/optimization.
- Engine returns suggestions; patch application/workflow automation is not included.
- Batch coordination is currently best-effort and falls back to per-file RL optimization when needed.
