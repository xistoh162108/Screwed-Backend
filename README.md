# Screwed-Backend

**NASA Space Apps Challenge 2024** - Agricultural Climate Simulation Game Backend

How to use: 
API: https://api.tedxkaist.org/api/v1/docs#

A FastAPI-based backend service that combines climate prediction ML models with LLM-driven natural language processing to create an educational farming simulation game. Players make agricultural decisions through natural language commands, and the system predicts crop yields based on NASA climate data.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [API Reference](#api-reference)
- [LLM Components](#llm-components)
- [ML Models](#ml-models)
- [Data Models](#data-models)
- [Game Scenarios & Flow](#game-scenarios--flow)
- [Setup & Deployment](#setup--deployment)
- [Development](#development)

---

## Overview

### Tech Stack

- **Framework**: FastAPI (Python 3.11)
- **Database**: PostgreSQL (hosted on GCP)
- **Task Queue**: Celery + Redis
- **LLM**: Google Gemini (via `google-generativeai`)
- **ML**: PyTorch (climate prediction) + LightGBM (yield prediction)
- **Deployment**: Docker Compose + nginx-proxy + Let's Encrypt

### Key Features

1. **Turn-based Timeline System**: Branching game timelines with optimistic concurrency control
2. **Natural Language Commands**: LLM-powered command parsing and validation
3. **Climate Prediction**: LSTM-based climate forecasting using NASA data (26 variables)
4. **Yield Prediction**: LightGBM models for crop yield estimation (corn, rice, wheat, soybean)
5. **Async Processing**: Celery workers handle long-running ML/LLM tasks
6. **Multi-timeline Support**: Players can fork timelines to explore different strategies

---

## System Architecture

```
┌─────────────┐
│   Unity     │ (Game Client)
│  Frontend   │
└──────┬──────┘
       │ HTTP/REST
       ▼
┌─────────────────────────────────────────────┐
│           FastAPI Backend                    │
│  ┌─────────────────────────────────────┐   │
│  │  API Layer (src/app/api/v1/)        │   │
│  │  - Turns, Commands, Outputs         │   │
│  │  - Sessions, Health                 │   │
│  └──────────────┬──────────────────────┘   │
│                 ▼                            │
│  ┌─────────────────────────────────────┐   │
│  │  Service Layer (src/app/services/)  │   │
│  │  - turn_service                     │   │
│  │  - command_service / command_task   │   │
│  │  - output_service / output_task     │   │
│  │  - prediction_service               │   │
│  └──────────────┬──────────────────────┘   │
│                 ▼                            │
│  ┌─────────────────────────────────────┐   │
│  │  Data Layer (src/app/models/)       │   │
│  │  - Turn, Command, Output, Session   │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
       │                        │
       ▼                        ▼
┌──────────────┐      ┌──────────────────┐
│  PostgreSQL  │      │  Redis + Celery  │
│   Database   │      │   Worker Queue   │
└──────────────┘      └──────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  External Services   │
                   │  - Google Gemini     │
                   │  - ML Models (local) │
                   └──────────────────────┘
```

### Request Flow

**Synchronous (CRUD operations)**:
```
Client → FastAPI Endpoint → Service Layer → Database → Response
```

**Asynchronous (LLM/ML tasks)**:
```
Client → FastAPI Endpoint → Service Layer → Celery Task (queued)
       ← Task ID returned

Celery Worker → LLM/ML Processing → Update Database → Store Result
```

---

## API Reference

All endpoints are prefixed with `/api/v1`

### 📊 Turns (Game Timeline Management)

Turns represent individual months in the game timeline with branching support.

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `POST` | `/turns` | Create new turn | `TurnCreate` | `TurnOut` |
| `GET` | `/turns/{turn_id}` | Get single turn | - | `TurnOut` |
| `GET` | `/turns` | List turns (filterable) | Query: `branch_id`, `parent_id`, `month`, `state`, `limit`, `cursor` | `List[TurnOut]` |
| `PATCH` | `/turns/{turn_id}/stats` | Update turn statistics | `TurnUpdateStats` | `TurnOut` |
| `POST` | `/turns/{turn_id}/state/{next_state}` | Transition turn state | - | `TurnOut` |
| `POST` | `/turns/{turn_id}/children` | Create child turn (same branch) | `TurnCreate` | `TurnOut` |
| `POST` | `/turns/{turn_id}/branch` | Create new branch | `TurnCreate` | `TurnOut` |
| `GET` | `/turns/{turn_id}/tree` | Get full turn tree | Query: `session_id`, `max_nodes`, `use_recursive_cte` | `TreeStructure` |
| `POST` | `/crash` | Handle concurrent turn creation | `CrashNodeInput` + Header: `If-Match` | `TurnOut` (200/201) or `ConflictOut` (409) |

**Turn States** (State Machine):
```
DRAFT → BRIEFED → COMMAND_PENDING → VALIDATING → VALIDATED →
COST_ESTIMATED → BUDGET_OK → SIMULATED → APPLIED
                    ↓
                REJECTED → COMMAND_PENDING (retry)
```

### 🎮 Commands (Player Instructions)

Commands are natural language instructions that players give each turn.

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `POST` | `/turns/{turn_id}/commands` | Create command (triggers async validation) | `CommandCreate` | `CommandCreateOut` |
| `GET` | `/turns/{turn_id}/commands` | List commands for turn | Query: `limit`, `cursor` | `List[CommandOut]` |
| `GET` | `/turns/{turn_id}/commands/{cmd_id}` | Get single command | - | `CommandOut` |
| `POST` | `/turns/{turn_id}/commands/{cmd_id}/validate` | Set validation result | `CommandValidateIn` | `CommandOut` |
| `POST` | `/turns/{turn_id}/commands/{cmd_id}/estimate-cost` | Set cost estimate | `CommandCostIn` | `CommandOut` |

**Example Command**:
```json
{
  "text": "옥수수 100에이커 심고 드립 관개 설치해줘",
  "payload": {}
}
```

### 📈 Outputs (Predictions & Results)

Outputs contain climate/yield predictions generated from commands.

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `POST` | `/commands/{command_id}/outputs` | Create output (triggers async prediction) | - | `OutputIdOut` |
| `GET` | `/outputs/{output_id}` | Get output | - | `OutputOut` |
| `GET` | `/turns/{turn_id}/outputs` | List outputs for turn | Query: `limit`, `cursor` | `List[OutputOut]` |
| `POST` | `/outputs/{output_id}/apply` | Apply output to game state | - | Status message |

**Output States**: `PENDING` → `RUNNING` → `COMPLETE` / `FAILED`

**Output Kinds**: `DENIED`, `ANSWER`, `PROCESSED`

### 🎯 Sessions (Game Sessions)

Sessions group multiple turns into a single gameplay session.

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `POST` | `/sessions` | Create session | `SessionCreate` | `SessionOut` |
| `GET` | `/sessions/{session_id}` | Get session | - | `SessionOut` |
| `GET` | `/sessions` | List sessions | Query: `limit`, `cursor` | `List[SessionOut]` |
| `PATCH` | `/sessions/{session_id}` | Update session | `SessionUpdate` | `SessionOut` |
| `DELETE` | `/sessions/{session_id}` | Delete session | - | 204 No Content |

### 🏥 Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |

---

## LLM Components

The system uses **Google Gemini** (`gemini-flash-latest`) for natural language processing with config-driven prompts.

### LLM Configuration Files

Located in [`src/app/utils/`](src/app/utils/):

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `normalizeUserinput.json` | Normalize typos, slang, extract action count | User text | `{normalized_text, action_count}` |
| `questionTypeChecker.json` | Classify input type | Normalized text | `{type: "Q"/"I"/"O"}` (Question/Instruction/Other) |
| `procedureAnalyzer.json` | Parse instruction into structured command | Instruction text | `{intent, crop, area, params...}` |
| `feedbackGenerator.json` | Generate narrative feedback | Simulation results | Natural language response |

### LLM Functions

Implemented in [`src/app/services/llm.py`](src/app/services/llm.py):

```python
# 1. Normalize user input
normalizeInput(message: str) -> dict
# Returns: {"normalized_text": str, "action_count": int}

# 2. Determine question type
determineQuestionType(normalizedData: dict) -> dict
# Returns: {"type": "Q" | "I" | "O"}

# 3. Analyze procedure (parse instruction)
procedureAnalyzer(sentence: str) -> dict
# Returns: {"intent": str, "crop": str, "target_area": str, ...}

# 4. Question handler (answer questions)
questionHandler(question_text: str) -> dict
# Returns: {"final_response": str, "status": "ANSWERED" | "ERROR"}

# 5. Generate feedback (after simulation)
generateFeedback(simulation_result: dict) -> str
```

### LLM Processing Flow

```
User Input: "옥수수 100에이커 심고 드립 관개 설치해줘"
    ↓
normalizeInput()
    ↓
{"normalized_text": "옥수수 100에이커 심고 드립 관개 설치.", "action_count": 2}
    ↓
determineQuestionType()
    ↓
{"type": "I"}  (Instruction)
    ↓
Check action_count
    ↓
VIOLATION: "한 턴에 하나의 조치만 가능합니다"
```

**Valid Single Action Example**:
```
User Input: "옥수수 100에이커 심어줘"
    ↓
normalizeInput() → {"normalized_text": "옥수수 100에이커 심기", "action_count": 1}
    ↓
determineQuestionType() → {"type": "I"}
    ↓
procedureAnalyzer() → {"intent": "심기", "crop": "옥수수", "area": 100, "unit": "에이커"}
    ↓
Execute simulation → Update game state
    ↓
generateFeedback() → "옥수수 100에이커를 성공적으로 심었습니다..."
```

---

## ML Models

### 1. Climate Prediction Model

**Location**: [`src/app/services/climate_inference.py`](src/app/services/climate_inference.py)

**Architecture**: FNN → ResLSTM Blocks → Attention Pooling → Gaussian Head

**Model Type**: PyTorch (CPU-only inference)

**Artifacts** (in [`artifacts/predictClimate/`](artifacts/predictClimate/)):
- `model.pt` (6.4 MB) - PyTorch model weights
- `normalizer_stats.json` (8.2 MB) - Feature normalization statistics
- `calibration.json` - Per-feature calibration parameters

**Input**:
- Recent L timesteps (default: 6 months) of F climate features
- Metadata: month, latitude, longitude
- NASA climate variables (26 features): T2M, RH2M, PRECTOTCORR, etc.

**Output**:
- Gaussian predictions: `[mu, logvar]` for each climate variable
- Calibrated using `log_t_f` and `scalar_s` from calibration.json

**Model Class**:
```python
class FNN_ResLSTM_AttnHead(nn.Module):
    - Pre-FNN: Linear → GELU → Dropout → Linear
    - ResLSTM Blocks (3 layers): LSTM with residual connections
    - Attention Pooling: Self-attention over timesteps
    - Gaussian Head: Output [mu, logvar] for uncertainty estimation
```

### 2. Yield Prediction Models

**Location**: [`src/app/services/yield_inference.py`](src/app/services/yield_inference.py)

**Model Type**: LightGBM (gradient boosting)

**Crops Supported**: Corn (maize), Rice, Wheat, Soybean

**Artifacts** (in [`artifacts/yieldInference/{crop}/`](artifacts/yieldInference/)):
- Each crop folder contains:
  - `model.pt` - LightGBM booster (PyTorch serialized)
  - `feature_names.json` - Expected input features

**Input Features** (generated by [`featurizer_yield.py`](src/app/services/featurizer_yield.py)):
- Climate statistics (mean, std, min, max for temperature, precipitation, humidity)
- Location metadata (latitude, longitude, region)
- Crop-specific parameters (planting date, variety)

**Output**:
- Crop yield prediction (tons per hectare)
- Uncertainty estimates (optional)

**Usage**:
```python
from app.services.yield_inference import YieldPredictorLGBM

predictor = YieldPredictorLGBM("artifacts/yieldInference/maize")
predictions = predictor.predict(feature_dataframe)
```

### 3. Prediction Service Integration

**Location**: [`src/app/services/prediction_service.py`](src/app/services/prediction_service.py)

Orchestrates climate + yield predictions:

```python
def run_prediction(
    variables: List[str],
    location: Dict[str, float],
    horizon_days: int,
    context: Dict[str, Any],
    user_text: str | None
) -> Tuple[Dict, Dict, Dict]:
    # Returns: (prediction, impact, delta_stats)
```

**Current Implementation**: Dummy/placeholder that generates synthetic data. Replace with actual climate/yield model calls.

---

## Data Models

### Database Schema

**Turn** ([`src/app/models/turn.py`](src/app/models/turn.py)):
```python
{
  id: str              # t_{uuid}
  parent_id: str       # Foreign key to parent turn
  branch_id: str       # Timeline branch identifier
  session_id: str      # Foreign key to session
  month: str           # YYYY-MM format
  state: TurnState     # State machine enum
  stats: JSONB         # {climate, yield, env, money, notes}
  created_at: datetime
  updated_at: datetime
}
```

**Command** ([`src/app/models/command.py`](src/app/models/command.py)):
```python
{
  id: str              # c_{uuid}
  turn_id: str         # Foreign key to turn
  text: str            # Natural language instruction
  validity: JSONB      # {is_valid, score, reasons[]}
  cost: JSONB          # {estimate, currency, breakdown[]}
  payload: JSON        # Additional metadata
  created_at: str
}
```

**Output** ([`src/app/models/output.py`](src/app/models/output.py)):
```python
{
  id: str              # o_{uuid}
  command_id: str      # Foreign key to command
  turn_id: str         # Foreign key to turn
  state: str           # PENDING/RUNNING/COMPLETE/FAILED
  kind: str            # DENIED/ANSWER/PROCESSED
  answer: str          # Natural language response
  impact: JSONB        # {top_features: [{name, contrib}]}
  prediction: JSONB    # {target, yhat[], ts_start, freq, ...}
  delta_stats: JSONB   # {baseline, pred_mean, delta}
  models: JSONB        # {climate_model, yield_model}
  assumptions: JSON[]  # Model assumptions
  denied_reasons: JSON[]
  created_at: str
  completed_at: str
}
```

**Session** ([`src/app/models/session.py`](src/app/models/session.py)):
```python
{
  id: str              # s_{uuid}
  title: str
  root_turn_id: str    # Reference to first turn
  meta: JSONB          # Custom metadata
  created_at: datetime
  updated_at: datetime
}
```

### Pydantic Schemas

Request/response schemas in [`src/app/schemas/`](src/app/schemas/):

- `TurnCreate`, `TurnOut`, `TurnUpdateStats`, `TurnState`
- `CommandCreate`, `CommandOut`, `CommandValidateIn`, `CommandCostIn`
- `OutputCreateIn`, `OutputOut`, `OutputIdOut`
- `SessionCreate`, `SessionUpdate`, `SessionOut`
- `CrashNodeInput`, `ConflictOut` (for crash handling)

---

## Game Scenarios & Flow

### Typical Gameplay Loop

```
1. Client creates session
   POST /sessions → {session_id: "s_abc123"}

2. Client creates initial turn
   POST /turns → {id: "t_root_b_xyz", state: "DRAFT", month: "2024-01"}

3. Player enters command
   POST /turns/t_root_b_xyz/commands
   Body: {text: "옥수수 100에이커 심어줘"}
   → Returns: {command_id: "c_def456", output_id: "o_ghi789", task_id: "celery-task-id"}

4. System processes command asynchronously
   - Celery worker runs command_task
   - LLM validates command (normalizeInput → determineQuestionType → procedureAnalyzer)
   - Updates command.validity in database
   - Triggers output generation

5. System generates prediction asynchronously
   - Celery worker runs output_task
   - Runs climate prediction (climate_inference.py)
   - Runs yield prediction (yield_inference.py)
   - Stores results in output.prediction, output.impact, output.delta_stats

6. Client polls for completion
   GET /outputs/o_ghi789
   → {state: "COMPLETE", kind: "PROCESSED", prediction: {...}, impact: {...}}

7. Client applies output
   POST /outputs/o_ghi789/apply
   → Updates turn state to APPLIED

8. Client creates next turn
   POST /turns/t_root_b_xyz/children
   Body: {month: "2024-02"}
   → {id: "t_child_1", parent_id: "t_root_b_xyz"}

9. Repeat from step 3
```

### Branching Scenario

```
Timeline A:
t_root → t_child_1 → t_child_2 (player: plant corn)
                         ↓
                    Player wants to try wheat instead
                         ↓
                    POST /turns/t_child_1/branch
                         ↓
Timeline B:          t_branch_1 (player: plant wheat)
```

### Crash Handling (Optimistic Concurrency)

```
Scenario: Two clients try to extend same timeline simultaneously

Client 1                          Client 2
   |                                 |
   POST /crash                       POST /crash
   current_id: t_xyz                 current_id: t_xyz
   next: {...}                       next: {...}
   |                                 |
   ✓ Returns 201 (created)           ✗ Returns 409 (conflict)
   t_new_1                           {latest_id: t_new_1, latest_etag: "..."}
                                     |
                                     Client decides: FORK or RETRY
                                     |
                                     POST /crash (with resolution: FORK)
                                     |
                                     ✓ Returns 201 (new branch created)
```

### Command Validation Scenarios

**Valid Single Action**:
```
Input: "옥수수 100에이커 심기"
→ action_count: 1
→ procedureAnalyzer: {intent: "심기", crop: "옥수수", area: 100}
→ Status: VALIDATED
→ Execute simulation
```

**Invalid Multiple Actions**:
```
Input: "옥수수 심고 밀 수확해줘"
→ action_count: 2
→ Status: VIOLATION
→ Response: "한 턴에 하나의 조치만 가능합니다"
```

**Invalid Delegation**:
```
Input: "알아서 해줘"
→ procedureAnalyzer: {intent: "위임_불가"}
→ Status: VIOLATION
→ Response: "구체적인 작물, 지역, 관개 방식을 지정해주세요"
```

**Question Handling**:
```
Input: "현재 수분 스트레스는 어때?"
→ questionType: "Q"
→ questionHandler: Query game state
→ Response: "현재 수분 스트레스 지수는 0.25입니다..."
```

---

## Setup & Deployment

### Prerequisites

- Docker & Docker Compose
- PostgreSQL database (or use provided GCP instance)
- Google Gemini API key

### Environment Variables

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/dbname

# Celery/Redis
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# LLM
GOOGLE_API_KEY=your-gemini-api-key

# Deployment (optional)
VIRTUAL_HOST=api.yourdomain.com
LETSENCRYPT_HOST=api.yourdomain.com
LETSENCRYPT_EMAIL=your-email@example.com
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d --build

# Services started:
# - backend (FastAPI on port 8000)
# - worker (Celery worker)
# - redis (message broker)
# - flower (Celery monitoring on port 5555)
# - nginx-proxy (reverse proxy on port 80/443)
# - letsencrypt (SSL certificate management)

# View logs
docker-compose logs -f backend
docker-compose logs -f worker

# Stop services
docker-compose down
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run backend
PYTHONPATH=./src uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run Celery worker (in separate terminal)
PYTHONPATH=./src celery -A app.core.celery_app worker -Q validation -l INFO

# Run Redis (or use Docker)
docker run -d -p 6379:6379 redis:7
```

### Accessing Services

- **API Docs**: http://localhost:8000/api/v1/docs
- **Flower (Celery Monitor)**: http://localhost:5555
- **Production API**: https://api.tedxkaist.org/api/v1/docs
- **Production Flower**: https://flower.api.tedxkaist.org

---

## Development

### Project Structure

```
.
├── src/
│   ├── app/
│   │   ├── api/v1/          # API endpoints
│   │   │   ├── turns.py
│   │   │   ├── commands.py
│   │   │   ├── outputs.py
│   │   │   ├── sessions.py
│   │   │   └── health.py
│   │   ├── core/            # Configuration
│   │   │   ├── config.py
│   │   │   └── celery_app.py
│   │   ├── db/              # Database
│   │   │   ├── base.py
│   │   │   └── session.py
│   │   ├── models/          # SQLAlchemy models
│   │   │   ├── turn.py
│   │   │   ├── command.py
│   │   │   ├── output.py
│   │   │   └── session.py
│   │   ├── schemas/         # Pydantic schemas
│   │   ├── services/        # Business logic
│   │   │   ├── turn_service.py
│   │   │   ├── command_service.py
│   │   │   ├── command_task.py      # Celery task
│   │   │   ├── output_service.py
│   │   │   ├── output_task.py       # Celery task
│   │   │   ├── prediction_service.py
│   │   │   ├── prediction_task.py   # Celery task
│   │   │   ├── llm.py               # LLM integration
│   │   │   ├── climate_inference.py # Climate ML model
│   │   │   ├── yield_inference.py   # Yield ML model
│   │   │   └── featurizer_yield.py
│   │   ├── utils/           # LLM config files
│   │   │   ├── normalizeUserinput.json
│   │   │   ├── questionTypeChecker.json
│   │   │   ├── procedureAnalyzer.json
│   │   │   └── feedbackGenerator.json
│   │   └── main.py          # FastAPI app
│   └── migrations/          # Database migrations
├── artifacts/               # ML model artifacts
│   ├── predictClimate/
│   │   ├── model.pt
│   │   ├── normalizer_stats.json
│   │   └── calibration.json
│   └── yieldInference/
│       ├── maize/
│       ├── rice/
│       ├── soybean/
│       └── wheat/
├── ml/                      # ML training scripts
│   ├── LMmodel.py
│   └── data_flaten.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

### Adding New Features

**1. New API Endpoint**:
```bash
# 1. Create schema (src/app/schemas/foo.py)
# 2. Create model (src/app/models/foo.py)
# 3. Create service (src/app/services/foo_service.py)
# 4. Create endpoint (src/app/api/v1/foo.py)
# 5. Register in src/app/api/v1/__init__.py
```

**2. New Celery Task**:
```bash
# 1. Create task file (src/app/services/my_task.py)
# 2. Add to celery.conf.imports in src/app/core/celery_app.py
# 3. Decorate function with @celery.task()
```

**3. New LLM Prompt**:
```bash
# 1. Create config JSON in src/app/utils/
# 2. Add function in src/app/services/llm.py using _call_gemini_model()
```

### Database Migrations

Currently using auto-creation via `Base.metadata.create_all()`. For production, set up Alembic:

```bash
# Initialize Alembic (if not done)
alembic init migrations

# Create migration
alembic revision --autogenerate -m "description"

# Apply migration
alembic upgrade head
```

### Testing

```bash
# Run tests (add pytest to requirements.txt)
pytest src/app/services/test_climate_inference.py
pytest src/app/services/test_yield_inference.py
```

---

## API Examples

### Create Session and Play

```bash
# 1. Create session
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "My First Game"}'

# Response: {"id": "s_abc123", "title": "My First Game", ...}

# 2. Create initial turn
curl -X POST http://localhost:8000/api/v1/turns \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s_abc123", "month": "2024-01"}'

# Response: {"id": "t_xyz789", "state": "DRAFT", ...}

# 3. Create command
curl -X POST http://localhost:8000/api/v1/turns/t_xyz789/commands \
  -H "Content-Type: application/json" \
  -d '{"text": "옥수수 100에이커 심기"}'

# Response: {"command_id": "c_def456", "output_id": "o_ghi789", "task_id": "..."}

# 4. Check output status
curl http://localhost:8000/api/v1/outputs/o_ghi789

# Response: {"state": "COMPLETE", "prediction": {...}, "impact": {...}}

# 5. Apply output
curl -X POST http://localhost:8000/api/v1/outputs/o_ghi789/apply

# 6. Create next turn
curl -X POST http://localhost:8000/api/v1/turns/t_xyz789/children \
  -H "Content-Type: application/json" \
  -d '{"month": "2024-02"}'
```

---

## Contributors

NASA Space Apps Challenge 2024 Team

## License

[Specify license]

---

## Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Celery Docs**: https://docs.celeryq.dev/
- **Google Gemini API**: https://ai.google.dev/docs
- **PyTorch**: https://pytorch.org/
- **LightGBM**: https://lightgbm.readthedocs.io/
