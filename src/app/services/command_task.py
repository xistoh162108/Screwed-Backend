# app/services/command_task.py
from app.core.celery_app import celery
from app.db.session import SessionLocal
from app.services import llm

@celery.task(name="app.services.command_task.validate_command_task")
def validate_command_task(command_id: str, output_id: str):
    from app.services import output_service as outsvc
    from app.models import Command as CommandModel
    db = SessionLocal()
    try:
        outsvc.set_running(db, output_id)

        # 1) DB에서 command 텍스트 로딩
        cmd = db.get(CommandModel, command_id)
        if not cmd:
            outsvc.set_failed(db, output_id, reasons=[f"Command not found: {command_id}"])
            return

        # 2) LLM 클라이언트 생성
        client = llm.create_client()

        # 3) 질문 유형 분류(= 간단한 유효성 분석 예시)
        classification = llm.determine_question_type(client, cmd.text)
        # 예: {"type": "schedule", "confidence": 0.92, ...}
        is_valid = classification.get("type", "unknown") != "unknown"
        reasons = [f"classification={classification}"]
        assumptions = ["validator:v1", "gemini:flash-latest"]

        if is_valid:
            # 필요 시 추가로 답변 생성 예시:
            # answer = llm.get_response(client, cmd.text)
            outsvc.set_complete_answer(
                db,
                output_id=output_id,
                answer_text="Command is valid.",   # answer 로 교체 가능
                models_meta={"validator": "gemini-flash-latest"},
                assumptions=assumptions,
            )
        else:
            outsvc.set_complete_denied(
                db,
                output_id=output_id,
                reasons=reasons or ["invalid command"],
                assumptions=assumptions,
            )

    except Exception as e:
        outsvc.set_failed(db, output_id, reasons=[str(e)])
        raise
    finally:
        db.close()