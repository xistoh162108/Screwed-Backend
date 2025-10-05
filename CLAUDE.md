# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NASA Space Apps Challenge project: A FastAPI backend for agricultural game simulation that combines climate prediction models with LLM-driven scenario generation. Players make farming decisions (crops, irrigation, fertilizer) and receive yield predictions based on NASA climate data (26 variables) processed through LSTM models.

## Development Commands

### Docker Environment
```bash
# Start all services (backend, worker, redis, flower, nginx-proxy, letsencrypt)
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f worker

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

### Local Development
```bash
# Run FastAPI server locally (set PYTHONPATH)
PYTHONPATH=./src uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run Celery worker locally
PYTHONPATH=./src celery -A app.core.celery_app worker -Q validation -l INFO

# Access API docs
# http://localhost:8000/api/v1/docs
```

### Database
- Uses PostgreSQL hosted at 34.64.87.169:5432 (database: screwed)
- Tables auto-created via `Base.metadata.create_all(bind=engine)` in [src/app/api/v1/turns.py:14](src/app/api/v1/turns.py#L14)
- Migrations directory exists at [src/migrations/](src/migrations/) but migrations are not currently configured with Alembic
- Connection configured in [src/app/db/session.py](src/app/db/session.py)

## Architecture

### Request Flow

**Standard API Flow:**
1. Client → FastAPI endpoints ([src/app/api/v1/](src/app/api/v1/))
2. Service layer orchestrates business logic ([src/app/services/](src/app/services/))
3. Database access via SQLAlchemy ORM ([src/app/models/](src/app/models/))

**Async Task Flow (LLM/ML):**
1. API endpoint triggers Celery task
2. Task queued in Redis (`validation` queue)
3. Celery worker processes task asynchronously
4. Result stored in Redis backend, status updated in DB

### Core Domain Model: Turn-Based Timeline

The game uses a **branching timeline** architecture centered around "Turns":

- **Turn**: Represents a single month of gameplay with state machine progression
  - States: `DRAFT → BRIEFED → COMMAND_PENDING → VALIDATING → VALIDATED → COST_ESTIMATED → BUDGET_OK → SIMULATED → APPLIED`
  - Allowed transitions defined in [src/app/services/turn_service.py:21-32](src/app/services/turn_service.py#L21)

- **Session**: Logical grouping of turns for one gameplay session

- **Branch**: Enables forking timelines when players want to explore alternative decisions
  - Each turn has `branch_id` to track which timeline it belongs to
  - Crash detection in `/crash` endpoint prevents conflicts when multiple clients try to extend same branch

- **Command**: Player's natural language instructions attached to a turn (e.g., "plant 100 acres of corn with drip irrigation")
  - Validated asynchronously via LLM in [src/app/services/command_task.py](src/app/services/command_task.py)

- **Output**: Prediction results (climate forecast, yield estimate) attached to a turn
  - Generated asynchronously via ML inference in [src/app/services/output_task.py](src/app/services/output_task.py)

See [src/app/models/turn.py](src/app/models/turn.py), [src/app/models/command.py](src/app/models/command.py), [src/app/models/output.py](src/app/models/output.py)

### Service Layer Organization

**Turn Management:**
- [turn_service.py](src/app/services/turn_service.py): CRUD operations, state transitions, tree traversal, crash handling (optimistic concurrency control with ETag)

**Command Processing (LLM):**
- [command_service.py](src/app/services/command_service.py): Validation orchestration
- [command_task.py](src/app/services/command_task.py): Celery task for async LLM validation
- [llm.py](src/app/services/llm.py): Google Gemini integration with config-driven prompts from [src/app/utils/](src/app/utils/)
  - `normalizeInput()`: Normalize user text and extract action count
  - `getQuestionType()`: Classify question type
  - `analyzeProcedure()`: Extract structured farming actions from natural language
  - `generateFeedback()`: Generate narrative feedback

**Prediction Pipeline (ML):**
- [prediction_service.py](src/app/services/prediction_service.py): Dummy prediction logic (placeholder)
- [prediction_task.py](src/app/services/prediction_task.py): Celery task for async prediction
- [climate_inference.py](src/app/services/climate_inference.py): FNN + ResLSTM + Attention model for climate prediction
  - Uses PyTorch models in [artifacts/predictClimate/](artifacts/predictClimate/)
  - CPU-only inference, requires `model.pt`, `normalizer_stats.json`, `calibration.json`
- [yield_inference.py](src/app/services/yield_inference.py): LightGBM model for crop yield prediction
  - Uses artifacts in [artifacts/yieldInference/](artifacts/yieldInference/)
- [featurizer_yield.py](src/app/services/featurizer_yield.py): Feature engineering for yield model

**Output Management:**
- [output_service.py](src/app/services/output_service.py): CRUD operations
- [output_repo.py](src/app/services/output_repo.py): Database access layer
- [output_task.py](src/app/services/output_task.py): Celery task for async output generation

### API Endpoints

All routes prefixed with `/api/v1`:

- **Turns** ([turns.py](src/app/api/v1/turns.py)):
  - `POST /turns` - Create turn
  - `GET /turns/{turn_id}` - Get single turn
  - `GET /turns` - List turns (filter by branch_id, parent_id, month, state)
  - `PATCH /turns/{turn_id}/stats` - Update stats
  - `POST /turns/{turn_id}/state/{next_state}` - Transition state
  - `POST /turns/{turn_id}/children` - Create child (same branch)
  - `POST /turns/{turn_id}/branch` - Create branch
  - `GET /turns/{turn_id}/tree` - Get full tree (requires session_id)
  - `POST /crash` - Handle concurrent turn creation with conflict resolution

- **Commands** ([commands.py](src/app/api/v1/commands.py)):
  - `POST /commands` - Create command (triggers async validation)
  - `GET /commands/{command_id}` - Get command
  - `GET /commands` - List commands (filter by turn_id)

- **Outputs** ([outputs.py](src/app/api/v1/outputs.py)):
  - `POST /outputs` - Create output (triggers async prediction)
  - `GET /outputs/{output_id}` - Get output
  - `GET /outputs` - List outputs (filter by turn_id, type)

- **Sessions** ([sessions.py](src/app/api/v1/sessions.py)):
  - `POST /sessions` - Create session
  - `GET /sessions/{session_id}` - Get session

- **Health** ([health.py](src/app/api/v1/health.py)):
  - `GET /health` - Health check

### Configuration

- [src/app/core/config.py](src/app/core/config.py): Pydantic settings (DATABASE_URL, CELERY_BROKER_URL, etc.)
- Environment variables loaded from `.env` file
- GOOGLE_API_KEY required for LLM features

### Celery Tasks

Celery app configured in [src/app/core/celery_app.py](src/app/core/celery_app.py):
- Queue: `validation`
- Broker/Backend: Redis (redis://redis:6379/0)
- Registered tasks: `output_task`, `command_task`, `prediction_task`

Task patterns:
- All tasks return JSON-serializable results
- Task IDs stored in database for status tracking
- Use `task_acks_late=True` and `visibility_timeout=3600` for reliability

### Machine Learning Models

**Climate Inference** ([climate_inference.py](src/app/services/climate_inference.py)):
- Input: Last L timesteps (default 6) of F features + (month, lat, lon)
- Output: Gaussian predictions (mean, variance) for climate variables
- Model: FNN → ResLSTM blocks → Attention pooling → Gaussian head
- Calibration applied per-feature using `calibration.json`

**Yield Inference** ([yield_inference.py](src/app/services/yield_inference.py)):
- Input: Climate features + crop/location metadata
- Output: Crop yield predictions
- Model: LightGBM (stored as joblib pickle)

Both models stored in [artifacts/](artifacts/) directory.

## Code Patterns

### Adding New Endpoints

1. Create router in [src/app/api/v1/](src/app/api/v1/)
2. Define schemas in [src/app/schemas/](src/app/schemas/)
3. Implement service logic in [src/app/services/](src/app/services/)
4. Register router in [src/app/api/v1/__init__.py](src/app/api/v1/__init__.py)

Example:
```python
# src/app/api/v1/foo.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services import foo_service
from app.schemas import FooCreate, FooOut

router = APIRouter(prefix="/foos", tags=["foos"])

@router.post("", response_model=FooOut)
def create_foo(body: FooCreate, db: Session = Depends(get_db)):
    return foo_service.create_foo(db, body)
```

### Adding Celery Tasks

1. Create task module in [src/app/services/](src/app/services/) (e.g., `my_task.py`)
2. Import `celery` from [src/app/core/celery_app.py](src/app/core/celery_app.py)
3. Decorate function with `@celery.task()`
4. Register in `celery.conf.imports` in [celery_app.py:17-20](src/app/core/celery_app.py#L17)

Example:
```python
# src/app/services/my_task.py
from app.core.celery_app import celery

@celery.task(bind=True, name="my_task.process")
def process_task(self, data: dict):
    # async processing logic
    return {"status": "done"}
```

### State Machine Transitions

Always use [turn_service.move_state()](src/app/services/turn_service.py) for state changes. Valid transitions defined in `ALLOWED_NEXT` dict. Raises `ValueError` for invalid transitions.

### Database Sessions

Use FastAPI dependency injection for DB sessions:
```python
def my_endpoint(db: Session = Depends(get_db)):
    # db is automatically closed after request
```

## Environment Variables

Required:
- `DATABASE_URL` - PostgreSQL connection string
- `CELERY_BROKER_URL` - Redis URL for Celery broker
- `CELERY_RESULT_BACKEND` - Redis URL for Celery results
- `GOOGLE_API_KEY` - Google Gemini API key

Optional:
- `VIRTUAL_HOST` - For nginx-proxy reverse proxy
- `LETSENCRYPT_HOST` - For Let's Encrypt SSL
- `LETSENCRYPT_EMAIL` - For Let's Encrypt notifications

## Monitoring

- **Flower**: Celery task monitor at `flower.api.tedxkaist.org` (or localhost:5555 locally)
- **FastAPI Docs**: Interactive API documentation at `/api/v1/docs`
- **Logs**: `docker-compose logs -f [service_name]`
