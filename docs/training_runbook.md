# Velox Optimization Engine - Complete Operations Guide

**Table of Contents**
- [System Requirements](#system-requirements)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [LLM Configuration](#llm-configuration)
- [Dataset Generation](#dataset-generation)
- [Model Pretraining](#model-pretraining)
- [Online Training & Feedback](#online-training--feedback)
- [Local Codebase Optimization](#local-codebase-optimization)
- [GitHub Repository Optimization](#github-repository-optimization)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware
- **CPU:** 4+ cores recommended for parallel optimization
- **RAM:** 8GB minimum, 16GB recommended
- **Disk:** 10GB free space (for models, datasets, database)
- **GPU:** Optional (CPU training works fine for <100k transitions)

### Software
- **Python:** 3.12 or higher
- **PostgreSQL:** 12+ (for optimization records, episodes, learned patterns)
- **Redis:** 6+ (for caching, replay buffers, rate limiting)
- **Git:** For repository cloning features
- **Optional:** Ollama (for local LLM fallback)

### Operating Systems
-  Windows 10/11
-  Linux (Ubuntu 20.04+, RHEL 8+)
-  macOS 12+

---

## Installation & Setup

### 1. Clone Repository

```powershell
git clone <repository-url>
cd CodeOptimizer
```

### 2. Create Python Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r code/code_optimizer_ai/requirements.txt
```

**Key packages installed:**
- `fastapi` — Web framework
- `uvicorn` — ASGI server
- `torch` — Deep learning (DQN)
- `langchain` + `langchain-openai` — LLM integration
- `psycopg2-binary` — PostgreSQL driver
- `redis` — Redis client
- `numpy`, `gymnasium` — RL utilities

### 4. Configure PostgreSQL

**Create database:**
```sql
CREATE DATABASE code_optimizer;
CREATE USER code_optimizer_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE code_optimizer TO code_optimizer_user;
```

**Test connection:**
```bash
psql -h localhost -U code_optimizer_user -d code_optimizer -c "SELECT version();"
```

### 5. Configure Redis

**Start Redis server:**
```bash
# Linux/macOS
redis-server --daemonize yes

# Windows (using WSL or native Redis)
redis-server
```

**Test connection:**
```bash
redis-cli ping
# Expected: PONG
```

### 6. Configure Environment Variables

**Create `.env` file:**
```bash
cp code/code_optimizer_ai/.env.example code/code_optimizer_ai/.env
```

**Edit `code/code_optimizer_ai/.env`:**
```bash
# Application
APP_NAME=VeloxOptimizer
APP_VERSION=1.0.0
DEBUG=False
TESTING=False

# Security
SECRET_KEY=<generate_with_openssl_rand_hex_32>
REQUIRE_AUTH_TOKEN=True
API_AUTH_TOKEN=<your_secure_token>
ENABLE_API_DOCS=False  # Set True for development

# Database
DATABASE_URL=postgresql://code_optimizer_user:your_secure_password@localhost:5432/code_optimizer
REDIS_URL=redis://localhost:6379/11
REDIS_KEY_PREFIX=code_optimizer_ai

# LLM Configuration (see LLM Configuration section below)
OPENROUTER_API_KEY=<your_openrouter_key>
OPENROUTER_PRIMARY_MODEL=openai/gpt-4o-mini
OPENROUTER_SECONDARY_MODEL=anthropic/claude-3.5-sonnet
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Security
ALLOWED_CODE_ROOTS=.
MAX_INLINE_CODE_CHARS=200000
MAX_UPLOAD_SIZE_BYTES=1000000
RATE_LIMIT_REQUESTS_PER_MINUTE=120

# Optimization
OBJECTIVE_RUNTIME_WEIGHT=0.5
OBJECTIVE_MEMORY_WEIGHT=0.5
DEFAULT_MAX_SUGGESTIONS=5
UNIT_TEST_COMMAND=pytest {path} --maxfail=1 -x

# Training
TRAINING_DATA_PATH=data/training
MODEL_CHECKPOINT_PATH=models/rl_policy
```

**Generate secure keys:**
```bash
# Generate SECRET_KEY
python -c "import secrets; print(secrets.token_hex(32))"

# Generate API_AUTH_TOKEN
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 7. Initialize Database Schema

```bash
python -c "
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from code.code_optimizer_ai.database.migrate import create_database_schema
asyncio.run(create_database_schema())
"
```

**Expected output:**
```
Created tables:
  • optimization_records
  • optimization_episodes
  • learned_patterns
```

### 8. Create Required Directories

```bash
mkdir -p data/training
mkdir -p models/rl_policy
mkdir -p data/training/jobs
```

---

## Running the Application

### Development Mode (with auto-reload)

```bash
python code/code_optimizer_ai/run.py
```

**Default settings:**
- Host: `0.0.0.0`
- Port: `8000`
- Workers: `1` (development)
- Reload: `True`

### Production Mode

**Using Python:**
```bash
python code/code_optimizer_ai/run.py --host 0.0.0.0 --port 8000 --workers 4 --no-reload
```

**Using shell script (Linux/macOS):**
```bash
chmod +x code/code_optimizer_ai/run.sh
./code/code_optimizer_ai/run.sh
```

**Using Uvicorn directly:**
```bash
uvicorn code.code_optimizer_ai.api_main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --no-access-log
```

### Running in Background (Linux/macOS)

```bash
nohup python code/code_optimizer_ai/run.py > logs/api.log 2>&1 &
echo $! > api.pid
```

**Stop server:**
```bash
kill $(cat api.pid)
```

### Health Check

```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-19T20:00:00Z",
  "version": "1.0.0"
}
```

### Verify API Access

**Without authentication:**
```bash
curl http://localhost:8000/
```

**With authentication:**
```bash
export API_TOKEN="your_token_from_env"
curl http://localhost:8000/health/details \
    -H "X-API-Token: $API_TOKEN"
```

---

## LLM Configuration

The system supports **three-tier LLM fallback**:
1. **Primary:** OpenRouter (GPT-4o-mini default)
2. **Secondary:** OpenRouter fallback model (Claude 3.5 Sonnet)
3. **Tertiary:** Local Ollama

### Option 1: OpenRouter (Recommended)

**1. Sign up at [OpenRouter.ai](https://openrouter.ai/)**

**2. Create API key**

**3. Configure in `.env`:**
```bash
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_PRIMARY_MODEL=openai/gpt-4o-mini
OPENROUTER_SECONDARY_MODEL=anthropic/claude-3.5-sonnet
LLM_TIMEOUT_SECONDS=45
```

**Supported models:**
- `openai/gpt-4o-mini` — Fast, cheap, good quality
- `openai/gpt-4o` — Higher quality, more expensive
- `anthropic/claude-3.5-sonnet` — Best reasoning
- `google/gemini-pro-1.5` — Good for code
- `meta-llama/llama-3.3-70b-instruct` — Open source

**Cost optimization:**
- Use `gpt-4o-mini` for primary (fast + cheap)
- Reserve `claude-3.5-sonnet` for complex analysis
- Set reasonable timeout to avoid hanging requests

### Option 2: Direct OpenAI

**Configure in `.env`:**
```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
DEFAULT_LLM_PROVIDER=openai
```

### Option 3: Local Ollama (Free, Offline)

**1. Install Ollama:**
```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

**2. Pull model:**
```bash
ollama pull llama3
# or
ollama pull codellama
ollama pull deepseek-coder
```

**3. Configure in `.env`:**
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

**4. Verify Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```

### LLM Provider Priority

The system tries providers in this order:
1. OpenRouter primary model
2. OpenRouter secondary model (if primary fails)
3. Ollama local model (if both OpenRouter attempts fail)

**To force Ollama-only mode:**
```bash
# Leave OPENROUTER_API_KEY unset
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

### Testing LLM Configuration

```bash
curl -X POST http://localhost:8000/optimize \
    -H "Content-Type: application/json" \
    -H "X-API-Token: $API_TOKEN" \
    -d '{
        "code": "def slow(): return sum([i**2 for i in range(1000000)])",
        "max_suggestions": 1
    }'
```

Check logs for LLM provider used:
```
INFO: LLM analysis completed using provider: openrouter
```

---

## Dataset Generation

### Generate Synthetic Training Data

**Purpose:** Create multi-step optimization episodes for offline DQN pretraining.

**Basic Usage:**
```bash
python code/code_optimizer_ai/generate_training_data.py \
    --output-jsonl data/training/pretrain_transitions.jsonl \
    --synthetic-samples 50000 \
    --seed 42
```

**Advanced Usage (with real episodes):**
```bash
python code/code_optimizer_ai/generate_training_data.py \
    --input-dir data/training \
    --output-jsonl data/training/pretrain_transitions.jsonl \
    --synthetic-samples 100000 \
    --seed 42 \
    --objective-runtime-weight 0.7 \
    --objective-memory-weight 0.3
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output-jsonl` | Required | Output JSONL file path |
| `--synthetic-samples` | 10000 | Number of synthetic transitions to generate |
| `--input-dir` | `data/training` | Directory with legacy episode JSON files |
| `--seed` | 42 | Random seed for reproducibility |
| `--objective-runtime-weight` | 0.5 | Runtime importance (0-1) |
| `--objective-memory-weight` | 0.5 | Memory importance (0-1) |
| `--reward-scale` | 10000.0 | Reward multiplier |

**Output Format (JSONL):**

Each line is a JSON object representing one transition:
```json
{
  "state": [0.85, 0.3, 0.6, 0.25, 0.4, 0.2, 0, 0, 0, 1, ...],
  "action_idx": 2,
  "reward": 8.5,
  "next_state": [0.72, 0.25, 0.65, 0.2, 0.35, 0.15, 0, 0, 1, 0, ...],
  "done": false
}
```

**Multi-Step Episodes:**

The generator creates episodes of 3-8 steps each:
- **Initial state:** Random complexity (0.5-0.95), derived metrics
- **Step actions:** Sampled from 25 action types
- **State evolution:** Complexity decreases, issues decay
- **Terminal flag:** Only last step has `done: true`
- **Diminishing returns:** Improvements scale down as complexity drops

**Expected Output:**
```
Generated 503 transitions from legacy episodes
Generated 50000 synthetic multi-step transitions
Total transitions: 50503
Wrote to: data/training/pretrain_transitions.jsonl
```

**Verify Output:**
```bash
# Count transitions
wc -l data/training/pretrain_transitions.jsonl

# Inspect first transition
head -n 1 data/training/pretrain_transitions.jsonl | python -m json.tool

# Check done flag distribution
grep '"done": true' data/training/pretrain_transitions.jsonl | wc -l
grep '"done": false' data/training/pretrain_transitions.jsonl | wc -l
```

---

## Model Pretraining

### Delete Old Checkpoint (if upgrading)

**Important:** The old checkpoint uses a deprecated architecture. Delete it before training:

```bash
rm -f models/rl_policy/dqn_model.pth
rm -f models/rl_policy/dqn_model_pretrain_summary.json
```

### Pretrain DQN (CLI)

**Basic Training (40 epochs, ~30 minutes on CPU):**
```bash
python code/code_optimizer_ai/pretrain_dqn.py \
    --train data/training/pretrain_transitions.jsonl \
    --output models/rl_policy/dqn_model.pth \
    --epochs 40 \
    --steps-per-epoch 1000 \
    --batch-size 64 \
    --shuffle \
    --fresh-start
```

**Training with Holdout Evaluation:**
```bash
# First, split dataset into train/holdout (80/20)
python -c "
import json, random
lines = open('data/training/pretrain_transitions.jsonl').readlines()
random.seed(42)
random.shuffle(lines)
split = int(0.8 * len(lines))
open('data/training/train.jsonl', 'w').writelines(lines[:split])
open('data/training/holdout.jsonl', 'w').writelines(lines[split:])
print(f'Train: {split}, Holdout: {len(lines) - split}')
"

# Train with holdout evaluation
python code/code_optimizer_ai/pretrain_dqn.py \
    --train data/training/train.jsonl \
    --holdout data/training/holdout.jsonl \
    --output models/rl_policy/dqn_model.pth \
    --epochs 40 \
    --steps-per-epoch 1000 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --gamma 0.99 \
    --epsilon-start 1.0 \
    --epsilon-end 0.05 \
    --shuffle \
    --fresh-start
```

**Training Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train` | Required | Path to training transitions JSONL |
| `--holdout` | None | Path to holdout JSONL (optional) |
| `--output` | `models/rl_policy/dqn_model.pth` | Output checkpoint path |
| `--epochs` | 40 | Number of training epochs |
| `--steps-per-epoch` | 1000 | Gradient updates per epoch |
| `--batch-size` | 64 | Replay buffer sample size |
| `--learning-rate` | 0.001 | Initial LR (anneals to 1e-5) |
| `--gamma` | 0.99 | Discount factor |
| `--epsilon-start` | 1.0 | Initial exploration rate |
| `--epsilon-end` | 0.05 | Final exploration rate |
| `--hidden-dim` | 256 | Q-network hidden layer size |
| `--replay-buffer-size` | 10000 | Replay buffer capacity |
| `--shuffle` | False | Shuffle transitions before training |
| `--fresh-start` | False | Ignore existing checkpoint |
| `--seed` | 42 | Random seed |

**Training Progress:**

```
INFO: Loaded 40000 training transitions
INFO: Loaded 10000 holdout transitions
INFO: Starting offline pretraining...

Epoch 1/40:
  - Updates: 1000
  - Avg loss: 1.853
  - Epsilon: 0.398
  
Epoch 10/40:
  - Updates: 10000
  - Avg loss: 0.823
  - Epsilon: 0.158

Epoch 20/40:
  - Updates: 20000
  - Avg loss: 0.521
  - Epsilon: 0.100

Epoch 40/40:
  - Updates: 40000
  - Avg loss: 0.234
  - Epsilon: 0.050

Holdout Evaluation:
  - Mean TD error: 0.421
  - Mean reward: 5.23
  - Action accuracy: 68.4%

Model saved to: models/rl_policy/dqn_model.pth
```

**Expected Output Files:**
- `models/rl_policy/dqn_model.pth` — Model weights + hyperparameters
- `models/rl_policy/dqn_model_pretrain_summary.json` — Training statistics

**Verify Model:**
```bash
# Check file exists and size
ls -lh models/rl_policy/dqn_model.pth

# Inspect training summary
cat models/rl_policy/dqn_model_pretrain_summary.json | python -m json.tool
```

### Pretrain DQN (API)


### Pretrain DQN (API)

**Start Training Job (Background):**
```bash
curl -X POST http://localhost:8000/train \
    -H "X-API-Token: $API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "synthetic_samples": 50000,
        "epochs": 40,
        "steps_per_epoch": 1000,
        "batch_size": 64,
        "holdout_fraction": 0.20,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "shuffle": true,
        "fresh_start": false,
        "seed": 42,
        "background": true
    }'
```

**Response:**
```json
{
  "training_id": "job_1739267123_a1b2c3d4",
  "status": "initiated",
  "message": "Training job started in background",
  "job_metadata": {
    "synthetic_samples": 50000,
    "epochs": 40,
    "steps_per_epoch": 1000
  }
}
```

**Check Job Status:**
```bash
curl http://localhost:8000/train/job_1739267123_a1b2c3d4 \
    -H "X-API-Token: $API_TOKEN"
```

**Status Response:**
```json
{
  "job_id": "job_1739267123_a1b2c3d4",
  "status": "completed",
  "phase": "published",
  "config": {...},
  "evaluation": {
    "new_model_td_error": 0.421,
    "new_model_action_accuracy": 0.684,
    "prev_model_td_error": 0.523,
    "prev_model_action_accuracy": 0.612,
    "td_error_delta_pct": -19.5,
    "gate_passed": true
  },
  "artifacts": {
    "model_checkpoint": "models/rl_policy/dqn_model.pth",
    "dataset_path": "data/training/dataset_job_1739267123_a1b2c3d4.jsonl",
    "train_split": "data/training/train_job_1739267123_a1b2c3d4.jsonl",
    "holdout_split": "data/training/holdout_job_1739267123_a1b2c3d4.jsonl"
  }
}
```

**Job Status Values:**

| Status | Description |
|--------|-------------|
| `initiated` | Job created, not yet started |
| `generating` | Generating synthetic transitions |
| `training` | DQN training in progress |
| `evaluating` | Running holdout evaluation |
| `publishing` | Copying model to production path |
| `completed` | Model published successfully |
| `rejected` | Evaluation gate failed, old model kept |
| `failed` | Error occurred during training |

**List Recent Jobs:**
```bash
curl http://localhost:8000/train?limit=10 \
    -H "X-API-Token: $API_TOKEN"
```

**Synchronous Training (wait for completion):**
```bash
curl -X POST http://localhost:8000/train \
    -H "X-API-Token: $API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "synthetic_samples": 10000,
        "epochs": 10,
        "steps_per_epoch": 500,
        "background": false
    }'
```
 **Warning:** This will block until training completes (may take 10-60 minutes).

---

## Online Training & Feedback

### Shadow Feedback Collection

**Purpose:** Collect real-world optimization outcomes to improve the model over time.

**How It Works:**

1. **User requests optimization** via API
2. **Active policy** recommends optimization
3. **Suggestion applied** (by user or system)
4. **Outcome measured:** actual runtime/memory improvement
5. **Feedback recorded** to `data/training/online_feedback.jsonl`
6. **Shadow policy** trains on this feedback in background

**Feedback Format:**
```json
{
  "state": [0.75, 0.3, 0.6, ...],
  "action_idx": 2,
  "reward": 8.5,
  "next_state": [0.65, 0.25, 0.62, ...],
  "done": true,
  "source": "shadow_feedback",
  "timestamp": "2026-02-19T20:30:15.123Z"
}
```

**Check Feedback Collection:**
```bash
# View recent feedback
tail -n 20 data/training/online_feedback.jsonl

# Count feedback records
wc -l data/training/online_feedback.jsonl

# Analyze reward distribution
python -c "
import json
rewards = [json.loads(line)['reward'] for line in open('data/training/online_feedback.jsonl')]
print(f'Total: {len(rewards)}')
print(f'Mean: {sum(rewards)/len(rewards):.2f}')
print(f'Min: {min(rewards):.2f}, Max: {max(rewards):.2f}')
"
```

### Retrain with Online Feedback

**Combine synthetic + real feedback:**
```bash
# Merge datasets
cat data/training/pretrain_transitions.jsonl \
    data/training/online_feedback.jsonl \
    > data/training/combined.jsonl

# Shuffle
shuf data/training/combined.jsonl > data/training/combined_shuffled.jsonl

# Retrain
python code/code_optimizer_ai/pretrain_dqn.py \
    --train data/training/combined_shuffled.jsonl \
    --output models/rl_policy/dqn_model.pth \
    --epochs 40 \
    --steps-per-epoch 1000 \
    --batch-size 64 \
    --shuffle
```

**Recommended Retraining Schedule:**

| Feedback Count | Action |
|----------------|--------|
| 0 - 100 | Use synthetic-only model |
| 100 - 1000 | Mix 90% synthetic, 10% real |
| 1000 - 10000 | Mix 70% synthetic, 30% real |
| 10000+ | Consider pure real feedback or 50/50 mix |

### Active/Shadow Policy Promotion

**Check policy performance:**
```bash
curl http://localhost:8000/metrics \
    -H "X-API-Token: $API_TOKEN"
```

**Response includes:**
```json
{
  "active_policy": {
    "model_version": "job_1739267123_a1b2c3d4",
    "avg_reward": 5.23,
    "success_rate": 0.87
  },
  "shadow_policy": {
    "model_version": "testing",
    "avg_reward": 6.15,
    "success_rate": 0.91
  }
}
```

**Manual policy swap (if shadow outperforms):**
```python
# In Python console or script
from code.code_optimizer_ai.ml.rl_optimizer import get_rl_optimizer

optimizer = get_rl_optimizer()
optimizer.policy_manager.promote_shadow_to_active()
print("Shadow policy promoted to active")
```

---

## Local Codebase Optimization

### Option 1: Optimize Inline Code

**Simple Example:**
```bash
curl -X POST http://localhost:8000/optimize \
    -H "X-API-Token: $API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "objective_weights": {"runtime": 0.8, "memory": 0.2},
        "max_suggestions": 3,
        "run_validation": true
    }'
```

**Response:**
```json
{
  "suggestions": [
    {
      "suggestion_id": "uuid-1234",
      "category": "caching_strategy",
      "priority": "high",
      "description": "Add memoization to eliminate exponential recursion",
      "original_code": "def fibonacci(n):...",
      "optimized_code": "from functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fibonacci(n):...",
      "patch_diff": "--- original\n+++ optimized\n@@ -1,4 +1,6 @@\n+from functools import lru_cache\n+\n+@lru_cache(maxsize=None)\n def fibonacci(n):...",
      "expected_runtime_delta_pct": -99.8,
      "expected_memory_delta_pct": 0.1,
      "expected_weighted_score": -79.86,
      "confidence": 0.87,
      "validation_status": "validated",
      "implementation_effort": "low",
      "reasoning": "Exponential recursion detected. Memoization will cache results and reduce complexity from O(2^n) to O(n)."
    }
  ],
  "processing_time_seconds": 2.3
}
```

### Option 2: Optimize File

**Optimize a specific file:**
```bash
curl -X POST http://localhost:8000/optimize/file \
    -H "X-API-Token: $API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "file_path": "test_samples/slow_sort.py",
        "objective_weights": {"runtime": 1.0, "memory": 0.0},
        "max_suggestions": 5,
        "run_validation": true,
        "unit_test_command": "pytest test_samples/test_slow_sort.py"
    }'
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `code` or `file_path` | string | Required | Code string or file path |
| `objective_weights` | object | `{runtime: 0.5, memory: 0.5}` | Optimization priorities |
| `max_suggestions` | int | 5 | Maximum suggestions to return |
| `run_validation` | bool | true | Enable syntax/test/benchmark validation |
| `unit_test_command` | string | `pytest {path}` | Test command (requires `ENABLE_API_UNIT_TEST_COMMAND_OVERRIDE=true`) |

### Option 3: Scan Directory (Batch)

**Scan multiple files without optimization:**
```bash
curl -X POST http://localhost:8000/scan \
    -H "X-API-Token: $API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "directory": "test_samples",
        "file_pattern": "*.py",
        "include_metrics": true
    }'
```

**Response:**
```json
{
  "scan_results": [
    {
      "file_path": "test_samples/slow_sort.py",
      "complexity_score": 85.0,
      "maintainability_score": 45.0,
      "bottlenecks": ["O(n^2) bubble sort algorithm"],
      "opportunities": ["Replace with built-in sorted() or quicksort"],
      "lines_of_code": 42,
      "functions": 3,
      "classes": 0
    },
    ...
  ],
  "summary": {
    "total_files": 5,
    "avg_complexity": 62.3,
    "high_complexity_files": 2
  }
}
```

### Option 4: Monitor Directory (Continuous)

**Start continuous monitoring:**
```bash
curl -X POST http://localhost:8000/monitor/directory \
    -H "X-API-Token: $API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "directory": "src",
        "interval_seconds": 300,
        "auto_optimize": false,
        "notification_webhook": "https://hooks.slack.com/services/..."
    }'
```

This starts a background agent that:
- Scans directory every N seconds
- Detects code changes
- Analyzes complexity/bottlenecks
- Optionally auto-optimizes
- Sends notifications

**Check monitor status:**
```bash
curl http://localhost:8000/pipeline/status \
    -H "X-API-Token: $API_TOKEN"
```

---

## GitHub Repository Optimization

### Scan Repository

**Scan without optimization:**
```bash
curl -X POST http://localhost:8000/scan/repository \
    -H "X-API-Token: $API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "repository_url": "https://github.com/owner/repo",
        "ref": "main",
        "max_files": 50,
        "min_complexity": 50.0
    }'
```

**Response:**
```json
{
  "repository": "owner/repo",
  "ref": "main",
  "scan_summary": {
    "total_files": 147,
    "python_files": 89,
    "scanned_files": 50,
    "avg_complexity": 58.3,
    "high_complexity_files": 12
  },
  "hotspots": [
    {
      "file_path": "src/algorithms/sort.py",
      "complexity_score": 92.0,
      "loc": 234,
      "bottlenecks": 8,
      "optimization_potential": "high"
    },
    ...
  ]
}
```

### Optimize Repository

**Full repository optimization:**
```bash
curl -X POST http://localhost:8000/optimize/repository \
    -H "X-API-Token: $API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "repository_url": "https://github.com/owner/repo",
        "ref": "main",
        "max_files": 20,
        "min_complexity": 60.0,
        "objective_weights": {"runtime": 0.7, "memory": 0.3},
        "max_suggestions_per_file": 3,
        "enable_cross_file_context": true,
        "enable_batch_coordination": false,
        "run_validation": false
    }'
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repository_url` | string | Required | GitHub HTTPS URL only |
| `ref` | string | `main` | Branch, tag, or commit SHA |
| `max_files` | int | 50 | Max files to analyze |
| `min_complexity` | float | 50.0 | Complexity threshold for inclusion |
| `objective_weights` | object | `{runtime: 0.5, memory: 0.5}` | Optimization priorities |
| `max_suggestions_per_file` | int | 5 | Suggestions per file |
| `enable_cross_file_context` | bool | false | Include import graph context |
| `enable_batch_coordination` | bool | false | Coordinate multi-file suggestions |
| `run_validation` | bool | false | Run validation (slow for repos) |

**Response:**
```json
{
  "repository": "owner/repo",
  "ref": "main",
  "clone_path": "/tmp/velox_clone_abc123",
  "optimization_results": [
    {
      "file_path": "src/algorithms/sort.py",
      "status": "success",
      "suggestions": [
        {
          "suggestion_id": "uuid-5678",
          "category": "algorithm_change",
          "description": "Replace bubble sort with Timsort (Python's built-in sorted)",
          "expected_runtime_delta_pct": -95.0,
          ...
        }
      ]
    },
    {
      "file_path": "src/utils/cache.py",
      "status": "success",
      "suggestions": [...]
    }
  ],
  "summary": {
    "total_files_processed": 20,
    "successful_optimizations": 18,
    "failed_optimizations": 2,
    "total_suggestions": 47,
    "avg_expected_improvement_pct": -42.5
  }
}
```

### Cross-File Context

**Enable import graph awareness:**
```json
{
  "repository_url": "https://github.com/owner/repo",
  "enable_cross_file_context": true
}
```

When enabled:
- Analyzes module import relationships
- Includes relevant imported code in LLM context
- Suggests optimizations that consider dependencies
- Example: "This function is imported by 5 other modules — optimization will cascade"

### Batch Coordination

**Enable multi-file coordinated suggestions:**
```json
{
  "repository_url": "https://github.com/owner/repo",
  "enable_batch_coordination": true
}
```

When enabled:
- Groups related files (e.g., model + view + controller)
- Sends all related code to LLM in one request
- Gets coordinated optimization suggestions across files
- Example: "Refactor data structure in model.py and update all 3 views to match"

---

## Troubleshooting

### Application Won't Start

**Problem:** `ModuleNotFoundError: No module named 'code.code_optimizer_ai'`

**Solution:**
```bash
# Ensure you're in the repository root
pwd  # Should show .../CodeOptimizer

# Verify virtual environment is activated
which python  # Should show .../venv/bin/python

# Reinstall dependencies
pip install -r code/code_optimizer_ai/requirements.txt
```

**Problem:** `psycopg2.OperationalError: could not connect to server`

**Solution:**
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Check connection string in .env
echo $DATABASE_URL

# Test connection manually
psql $DATABASE_URL -c "SELECT version();"
```

**Problem:** `redis.exceptions.ConnectionError: Error connecting to Redis`

**Solution:**
```bash
# Check Redis is running
redis-cli ping

# Check Redis URL in .env
echo $REDIS_URL

# Start Redis if not running
redis-server --daemonize yes
```

### Training Issues

**Problem:** `No transitions generated`

**Solution:**
```bash
# Ensure output directory exists
mkdir -p data/training

# Generate with explicit synthetic samples
python code/code_optimizer_ai/generate_training_data.py \
    --output-jsonl data/training/pretrain_transitions.jsonl \
    --synthetic-samples 50000
```

**Problem:** `Replay buffer too small (batch_size > transitions)`

**Solution:**
```bash
# Option 1: Generate more transitions
# Option 2: Reduce batch size
python code/code_optimizer_ai/pretrain_dqn.py \
    --train data/training/pretrain_transitions.jsonl \
    --batch-size 32  # Reduced from 64
```

**Problem:** `Evaluation gate rejected new model`

**Solution:**
```bash
# Check if model is actually worse
cat models/rl_policy/dqn_model_pretrain_summary.json

# Option 1: Force publish (skip gate)
python code/code_optimizer_ai/pretrain_dqn.py \
    --train data/training/pretrain_transitions.jsonl \
    --fresh-start  # Bypasses comparison

# Option 2: Improve training data quality
# Generate more diverse synthetic episodes
```

**Problem:** `Model not updating after /train`

**Solution:**
The API caches the model in memory. Restart to reload:
```bash
# Find process
ps aux | grep "code.code_optimizer_ai.api_main"

# Kill and restart
kill <pid>
python code/code_optimizer_ai/run.py
```

### LLM Issues

**Problem:** `LLM request timeout`

**Solution:**
```bash
# Increase timeout in .env
LLM_TIMEOUT_SECONDS=120

# Or switch to faster model
OPENROUTER_PRIMARY_MODEL=openai/gpt-4o-mini
```

**Problem:** `OpenRouter API key invalid`

**Solution:**
```bash
# Verify key in .env
echo $OPENROUTER_API_KEY

# Test key directly
curl https://openrouter.ai/api/v1/models \
    -H "Authorization: Bearer $OPENROUTER_API_KEY"

# Use Ollama as fallback
# Leave OPENROUTER_API_KEY blank to force Ollama mode
```

**Problem:** `Ollama connection refused`

**Solution:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Verify model is pulled
ollama list
ollama pull llama3
```

### Validation Issues

**Problem:** `Syntax validation failed on valid code`

**Solution:**
Check Python version compatibility:
```bash
python --version  # Must be 3.12+

# Disable validation temporarily
curl -X POST http://localhost:8000/optimize \
    -d '{"code": "...", "run_validation": false}'
```

**Problem:** `Unit test command override not allowed`

**Solution:**
```bash
# Enable in .env
ENABLE_API_UNIT_TEST_COMMAND_OVERRIDE=true

# Restart API
```

### Performance Issues

**Problem:** Repository optimization is very slow

**Solution:**
```bash
# Reduce max_files
# Disable validation for repo scans
# Use faster LLM model
curl -X POST http://localhost:8000/optimize/repository \
    -d '{
        "repository_url": "...",
        "max_files": 10,
        "run_validation": false
    }'
```

**Problem:** High memory usage during training

**Solution:**
```bash
# Reduce replay buffer size
python code/code_optimizer_ai/pretrain_dqn.py \
    --replay-buffer-size 5000  # Default: 10000

# Reduce batch size
python code/code_optimizer_ai/pretrain_dqn.py \
    --batch-size 32  # Default: 64
```

### Database Issues

**Problem:** `Table does not exist`

**Solution:**
```bash
# Run migration
python -c "
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from code.code_optimizer_ai.database.migrate import create_database_schema
asyncio.run(create_database_schema())
"
```

**Problem:** Redis key conflicts with other apps

**Solution:**
```bash
# Use unique prefix in .env
REDIS_KEY_PREFIX=velox_prod_

# Or use different Redis database number
REDIS_URL=redis://localhost:6379/15
```

---

## Advanced Topics

### Custom Action Types

To add new optimization types, edit:
`code/code_optimizer_ai/ml/training_semantics.py`

```python
PRODUCTION_ACTION_TYPES: List[str] = [
    "algorithm_change",
    "data_structure_optimization",
    # ... existing types ...
    "custom_ml_optimization",  # Your new type
    "no_change",
]
```

Then retrain the model (action space size changes from 25 to 26).

### Distributed Training

For large datasets (>500k transitions):

```bash
# Use multiple workers with gradient accumulation
# (Requires code modification — not built-in)

```

### Model Deployment

**Production deployment checklist:**
- [ ] Train on 50k-100k transitions
- [ ] Holdout action accuracy > 60%
- [ ] Test on unseen code samples
- [ ] Enable auth (`REQUIRE_AUTH_TOKEN=true`)
- [ ] Set rate limits appropriately
- [ ] Monitor with `/metrics` endpoint
- [ ] Collect shadow feedback
- [ ] Schedule periodic retraining (weekly/monthly)

---

## Quick Reference Commands

**Setup:**
```bash
# Install
pip install -r code/code_optimizer_ai/requirements.txt
cp code/code_optimizer_ai/.env.example code/code_optimizer_ai/.env
# Edit .env with your config

# Initialize database
python -c "import asyncio, sys; from pathlib import Path; sys.path.insert(0, '.'); from code.code_optimizer_ai.database.migrate import create_database_schema; asyncio.run(create_database_schema())"

# Create directories
mkdir -p data/training models/rl_policy
```

**Training:**
```bash
# Generate dataset (50k transitions)
python code/code_optimizer_ai/generate_training_data.py \
    --output-jsonl data/training/pretrain_transitions.jsonl \
    --synthetic-samples 50000

# Train model (40 epochs, ~30min CPU)
python code/code_optimizer_ai/pretrain_dqn.py \
    --train data/training/pretrain_transitions.jsonl \
    --output models/rl_policy/dqn_model.pth \
    --epochs 40 \
    --steps-per-epoch 1000 \
    --shuffle \
    --fresh-start
```

**Running:**
```bash
# Start API
python code/code_optimizer_ai/run.py

# Check health
curl http://localhost:8000/health

# Optimize code
curl -X POST http://localhost:8000/optimize \
    -H "X-API-Token: $API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"code": "def slow(): return sum([x**2 for x in range(1000000)])"}'
```

**Monitoring:**
```bash
# Check metrics
curl http://localhost:8000/metrics -H "X-API-Token: $API_TOKEN"

# View feedback
tail -f data/training/online_feedback.jsonl

# Check training jobs
curl http://localhost:8000/train -H "X-API-Token: $API_TOKEN"
```

---

## Additional Resources

- **Architecture Guide:** See system explanation in previous sections
- **API Documentation:** Enable with `ENABLE_API_DOCS=true`, visit `/docs`
- **Security Guide:** `SECURITY.md` in repository root
- **Development:** `CONTRIBUTING.md` for code contributions

---

**Last Updated:** February 19, 2026  
**Version:** 1.0.0
