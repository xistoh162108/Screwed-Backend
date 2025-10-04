# app/services/command_task.py
from app.core.celery_app import celery
from app.db.session import SessionLocal
from app.services import output_service

@celery.task(name="app.services.command_task.validate_command_task")
def validate_command_task(command_id: str, output_id: str):
    db = SessionLocal()
    try:
        output_service.set_running(db, output_id)

        # TODO: 여기에 LLM 또는 규칙 기반 검증 로직
        is_valid = True
        reasons = []
        assumptions = ["validator:v1"]

        if is_valid:
            output_service.set_complete_answer(
                db,
                output_id=output_id,
                answer_text="Command is valid.",
                models_meta={"validator": "rule-engine-0.1"},
                assumptions=assumptions,
            )
        else:
            output_service.set_complete_denied(
                db,
                output_id=output_id,
                reasons=reasons or ["invalid command"],
                assumptions=assumptions,
            )
    except Exception as e:
        output_service.set_failed(db, output_id, reasons=[str(e)])
        raise
    finally:
        db.close()