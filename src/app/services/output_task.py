# app/services/output_task.py
from app.core.celery_app import celery
from app.db.session import SessionLocal   # get_db_session 없으면 SessionLocal 직접 사용
from app.services.output_repo import (
    set_running, set_complete_answer, set_complete_processed, set_failed
)

def run_llm_and_postprocess(command_id: str) -> dict:
    # 실제 LLM 호출/후처리
    return {
        "kind": "ANSWER",
        "answer": f"Generated answer for command {command_id}",
        "models_meta": {"provider": "openai", "model": "gpt-4.1-mini"},
        "assumptions": ["assume X", "assume Y"],
        # "impact": {...}, "prediction": {...}, "delta_stats": {...}  # kind가 PROCESSED일 때 사용
    }

@celery.task(name="app.services.output_task.generate_output_task", bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3, time_limit=120)
def generate_output_task(self, output_id: str, command_id: str):
    db = SessionLocal()
    try:
        set_running(db, output_id)
        result = run_llm_and_postprocess(command_id)
        if result.get("kind") == "ANSWER":
            set_complete_answer(
                db, output_id,
                answer_text=result["answer"],
                models_meta=result.get("models_meta"),
                assumptions=result.get("assumptions"),
            )
        else:
            set_complete_processed(
                db, output_id,
                impact=result.get("impact"),
                prediction=result.get("prediction"),
                delta_stats=result.get("delta_stats"),
                models_meta=result.get("models_meta"),
                assumptions=result.get("assumptions"),
            )
        return {"status": "ok", "output_id": output_id}
    except Exception as e:
        set_failed(db, output_id, reasons=[str(e)])
        raise
    finally:
        db.close()