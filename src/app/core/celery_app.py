# app/core/celery_app.py
from celery import Celery
import os

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery = Celery("llm_worker", broker=BROKER_URL, backend=RESULT_BACKEND)
celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    worker_max_tasks_per_child=100,
    task_acks_late=True,
    broker_transport_options={"visibility_timeout": 3600},
    imports=("app.services.output_task",
              "app.services.command_task",
            ),
)

