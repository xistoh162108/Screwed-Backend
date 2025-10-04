# app/services/output_services.py
from sqlalchemy.orm import Session
from app.models import Output as OutputModel
from app.services.output_repo import create_from_command, get_output, list_by_turn
from app.services.output_task import generate_output_task

def enqueue_output_from_command(db: Session, command_id: str) -> str:
    out_id_obj = create_from_command(db, command_id)
    output_id = out_id_obj.output_id

    async_result = generate_output_task.delay(output_id, command_id)

    obj = db.get(OutputModel, output_id)
    if obj:
        models = obj.models or {}
        models["task_id"] = async_result.id
        obj.models = models
        db.add(obj); db.commit()

    return output_id