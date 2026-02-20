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

## Quick Start

### 1. Installation

```powershell
# Clone and setup
git clone <repository-url>
cd CodeOptimizer
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# or: source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r code/code_optimizer_ai/requirements.txt

# Configure environment
cp code/code_optimizer_ai/.env.example code/code_optimizer_ai/.env
# Edit .env with your configuration
```

### 2. Setup Services

**PostgreSQL:**
```sql
CREATE DATABASE code_optimizer;
CREATE USER code_optimizer_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE code_optimizer TO code_optimizer_user;
```

**Redis:**
```bash
redis-server --daemonize yes
```

**Initialize Database:**
```bash
python -c "import asyncio, sys; from pathlib import Path; sys.path.insert(0, '.'); from code.code_optimizer_ai.database.migrate import create_database_schema; asyncio.run(create_database_schema())"
```

### 3. Configure LLM (Choose One)

**Option A: OpenRouter (Recommended)**
```bash
# Get API key from https://openrouter.ai/
# Add to .env:
OPENROUTER_API_KEY=sk-or-v1-xxxxx
OPENROUTER_PRIMARY_MODEL=anthropic/claude-sonnet-4.5
```

**Option B: Local Ollama (Free)**
```bash
# Install Ollama from https://ollama.ai/
ollama run qwen3-coder
# Add to .env:
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-coder
```

### 4. Train Model

```bash
# Generate training data (50k transitions)
python code/code_optimizer_ai/generate_training_data.py \
    --output-jsonl data/training/pretrain_transitions.jsonl \
    --synthetic-samples 50000 \
    --seed 42

# Train DQN model (40 epochs, ~30 minutes)
python code/code_optimizer_ai/pretrain_dqn.py \
    --train data/training/pretrain_transitions.jsonl \
    --output models/rl_policy/dqn_model.pth \
    --epochs 40 \
    --steps-per-epoch 1000 \
    --shuffle \
    --fresh-start
```

### 5. Run Application

```bash
python code/code_optimizer_ai/run.py
```

**Verify:**
```bash
curl http://localhost:8000/health
```

### 6. Optimize Code

```bash
curl -X POST http://localhost:8000/optimize \
    -H "Content-Type: application/json" \
    -H "X-API-Token: your_token" \
    -d '{
        "code": "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "max_suggestions": 3
    }'
```

** For complete documentation, see [docs/training_runbook.md](docs/training_runbook.md)**

---

## Testing

**Run all tests:**
```bash
pytest code/code_optimizer_ai/tests/ -v
```

**Run specific test suites:**
```bash
# Integration tests
pytest code/code_optimizer_ai/tests/test_integration.py -q

# RL/Training tests
pytest code/code_optimizer_ai/tests/test_pretrain_dqn.py \
       code/code_optimizer_ai/tests/test_generate_training_data.py -q

# Core components
pytest code/code_optimizer_ai/tests/test_core_components.py -q
```

**Security tests:**
```bash
pytest code/code_optimizer_ai/tests/test_security_fixes.py -v
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [training_runbook.md](docs/training_runbook.md) | **Complete operations guide** - setup, training, optimization, troubleshooting |
| [design_document.md](docs/design_document.md) | Architecture and design decisions |
| `SECURITY.md` | Security best practices and vulnerability disclosure |
| API docs | Enable with `ENABLE_API_DOCS=true`, visit `/docs` |

---

## Key Features

###  Intelligent Optimization Strategy
- **Reinforcement Learning (DQN)** selects best optimization type from 25 categories
- **LLM semantic analysis** understands code intent and bottlenecks
- **Multi-objective optimization** balances runtime vs memory tradeoffs

###  Safe Validation
- **Syntax checking** via AST parsing
- **Unit test execution** in isolated sandbox
- **Micro-benchmarking** measures actual performance improvements
- **Diff generation** shows exact changes

###  Flexible Deployment
- **Inline code** optimization via API
- **File-based** optimization for local codebases
- **GitHub repository** scanning and optimization
- **Continuous monitoring** with background agents

###  Production-Ready
- **PostgreSQL** for persistent storage
- **Redis** for caching and replay buffers
- **Authentication** and rate limiting
- **Evaluation gates** prevent bad model deployments
- **Shadow/active policy** for safe model updates

---

## Architecture Highlights

**Dueling Double DQN:**
- Separates state value from action advantage
- Reduces Q-value overestimation bias
- Soft Polyak target updates (τ=0.005)
- Huber loss for gradient stability

**Multi-Step Episodes:**
- Episodes of 3-8 sequential optimizations
- State evolution mirrors real refactoring workflows
- Diminishing returns as code quality improves

**Online Learning:**
- Shadow feedback collection from production
- Automatic retraining pipeline integration
- Active/shadow policy promotion based on performance

---

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
