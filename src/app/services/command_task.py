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
        if status in ("COMPLETED","ANSWERED"):
            outsvc.set_complete_answer(
                db, output_id=output_id, answer_text=final, models_meta=models_meta
            )
            # ★ 예측 태스크 발사 (원하시는 파라미터로)
            # 제공해준 변수 리스트 기본 세트
            VARIABLES = [
                "ALLSKY_SFC_LW_DWN","ALLSKY_SFC_PAR_TOT","ALLSKY_SFC_SW_DIFF","ALLSKY_SFC_SW_DNI",
                "ALLSKY_SFC_SW_DWN","ALLSKY_SFC_UVA","ALLSKY_SFC_UVB","ALLSKY_SFC_UV_INDEX",
                "ALLSKY_SRF_ALB","CLOUD_AMT","CLRSKY_SFC_PAR_TOT","CLRSKY_SFC_SW_DWN",
                "GWETPROF","GWETROOT","GWETTOP","PRECTOTCORR","PRECTOTCORR_SUM","PS","QV2M",
                "RH2M","T2M","T2MDEW","T2MWET","T2M_MAX","T2M_MIN","T2M_RANGE","TOA_SW_DWN","TS"
            ]
            run_prediction_task.delay(
                output_id=output_id,
                variables=VARIABLES,
                location={"lat": 37.6, "lon": 127.0},  # TODO: 실제 좌표/필드에서 가져오기
                horizon_days=30,
                context={"answer": final},
            )

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