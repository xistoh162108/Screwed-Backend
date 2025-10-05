# app/workers/validate.py
from __future__ import annotations
from celery import shared_task
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Command as CommandModel
from app.services.output_service import (
    set_running,
    set_complete_answer,
    set_complete_denied,
    set_failed,
)
from app.services.command_utils import extract_user_text_from_command

# ⚠️ 경로를 당신의 파일 구조에 맞게! (지금 올린 코드가 들어 있는 모듈)
from app.services.ai_pipeline.event_handler import eventHandler

@shared_task(name="validate_command_task", queue="validation")
def validate_command_task(command_id: str, output_id: str):
    """
    Command → (Gemini 파이프라인 eventHandler) → Output 확정
    """
    db: Session = get_db()
    try:
        set_running(db, output_id)

        cmd = db.get(CommandModel, command_id)
        if not cmd:
            set_failed(db, output_id, reasons=[f"Command not found: {command_id}"])
            return

        user_input = extract_user_text_from_command(cmd)
        if not user_input:
            set_failed(db, output_id, reasons=["No user_input found in Command"])
            return

        # Gemini 파이프라인 실행
        result = eventHandler(user_input) or {}
        status = result.get("status")
        final_response = result.get("final_response") or "빈 응답"

        # 선택: 메타 기록(모델/파이프라인/추적 등)
        models_meta = {
            "engine": "google.gemini",
            "pipeline": "normalize → classify → (procedure/question) → feedback",
        }

        # 상태 매핑
        if status in ("COMPLETED", "ANSWERED"):
            set_complete_answer(
                db,
                output_id=output_id,
                answer_text=final_response,
                models_meta=models_meta,
            )
        elif status == "VIOLATION":
            set_complete_denied(
                db,
                output_id=output_id,
                reasons=[final_response],
                models_meta=models_meta,
            )
        elif status == "ERROR":
            set_failed(
                db,
                output_id=output_id,
                reasons=[final_response],
                models_meta=models_meta,
            )
        else:
            set_failed(
                db,
                output_id=output_id,
                reasons=[f"Unknown status from eventHandler: {status}"],
                models_meta=models_meta,
            )
    except Exception as e:
        set_failed(
            db,
            output_id=output_id,
            reasons=[f"Exception in validate_command_task: {e}"],
            models_meta={"stage": "worker-exception"},
        )