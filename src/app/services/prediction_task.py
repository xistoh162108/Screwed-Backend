# app/services/prediction_task.py (신규)
from __future__ import annotations
import logging
from typing import Dict, Any, List

from app.core.celery_app import celery
from app.db.session import SessionLocal
from app.services.output_service import set_prediction_only
from app.models import Output as OutputModel, Command as CommandModel
from app.services.prediction_service import run_prediction

logger = logging.getLogger(__name__)

@celery.task(name="app.services.prediction_task.run_prediction_task", queue="validation")
def run_prediction_task(
    *,
    output_id: str,
    variables: List[str],
    location: Dict[str, float] | None = None,
    horizon_days: int = 30,
    context: Dict[str, Any] | None = None,
):
    db = SessionLocal()
    try:
        out = db.get(OutputModel, output_id)
        if not out:
            logger.warning("run_prediction_task: output not found %s", output_id)
            return
        cmd = db.get(CommandModel, out.command_id) if out.command_id else None

        pred, impact, delta = run_prediction(
            variables=variables,
            location=location or {},
            horizon_days=horizon_days,
            context=context or {},
            user_text=(cmd.text if cmd else None),
        )

        set_prediction_only(
            db,
            output_id=output_id,
            prediction=pred,
            impact=impact,
            delta_stats=delta,
        )
        logger.info("Prediction updated for output_id=%s", output_id)
    except Exception as e:
        logger.exception("run_prediction_task error (output_id=%s): %s", output_id, e)
    finally:
        db.close()