# app/services/command_task.py
from __future__ import annotations

import logging
from app.core.celery_app import celery
from app.db.session import SessionLocal
from app.services import output_service as outsvc
from app.models import Command as CommandModel

# 당신이 만든 파이프라인 (경로가 다르면 맞게 변경)
from app.services.llm import eventHandler

logger = logging.getLogger(__name__)

@celery.task(name="app.services.command_task.validate_command_task", queue="validation")
def validate_command_task(command_id: str, output_id: str):
    """
    Command → (Gemini 파이프라인 eventHandler) → Output 확정

    상태 매핑 규칙:
      - COMPLETED / ANSWERED → set_complete_answer(kind=ANSWER)
      - VIOLATION            → set_complete_denied(kind=DENIED)
      - ERROR                → set_failed(state=FAILED)
      - 그 외/누락            → set_failed
    """
    db = SessionLocal()
    try:
        # RUNNING 전이
        outsvc.set_running(db, output_id)

        # Command 로드
        cmd = db.get(CommandModel, command_id)
        if not cmd:
            outsvc.set_failed(db, output_id, reasons=[f"Command not found: {command_id}"])
            logger.warning("validate_command_task: command not found (command_id=%s)", command_id)
            return

        user_text = cmd.text or ""
        if not user_text.strip():
            outsvc.set_failed(db, output_id, reasons=["Empty command text"])
            logger.warning("validate_command_task: empty text (command_id=%s)", command_id)
            return

        # 파이프라인 실행
        # eventHandler는 {"status": "...", "final_response": "..."} 형태를 기대
        result = eventHandler(user_text) or {}
        status = result.get("status")
        final_response = result.get("final_response") or "빈 응답"

        models_meta = {
            "engine": "google.gemini",
            "pipeline": "normalize→classify→(procedure|question)→feedback",
            "task": "validate_command_task",
        }

        # 상태 매핑
        if status in ("COMPLETED", "ANSWERED"):
            outsvc.set_complete_answer(
                db,
                output_id=output_id,
                answer_text=final_response,
                models_meta=models_meta,
                # 필요 시 assumptions/impact/prediction/delta_stats 확장 가능
            )
            logger.info("Output %s → COMPLETE/ANSWER", output_id)

        elif status == "VIOLATION":
            outsvc.set_complete_denied(
                db,
                output_id=output_id,
                reasons=[final_response],
                models_meta=models_meta,
            )
            logger.info("Output %s → COMPLETE/DENIED", output_id)

        elif status == "ERROR":
            outsvc.set_failed(
                db,
                output_id=output_id,
                reasons=[final_response],
                models_meta=models_meta,
            )
            logger.error("Output %s → FAILED (ERROR from pipeline)", output_id)

        else:
            outsvc.set_failed(
                db,
                output_id=output_id,
                reasons=[f"Unknown status from eventHandler: {status}"],
                models_meta=models_meta,
            )
            logger.error("Output %s → FAILED (unknown status=%s)", output_id, status)

    except Exception as e:
        # 시스템 예외는 FAILED로 마무리
        outsvc.set_failed(
            db,
            output_id=output_id,
            reasons=[f"Exception in validate_command_task: {e}"],
            models_meta={"stage": "worker-exception"},
        )
        logger.exception("validate_command_task exception (command_id=%s, output_id=%s)", command_id, output_id)
        raise
    finally:
        db.close()