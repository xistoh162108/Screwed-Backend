from celery import Celery

celery = Celery(
    "llm_worker",
    broker="redis://localhost:6379/0",      # 환경에 맞게 수정
    backend="redis://localhost:6379/1",
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    worker_max_tasks_per_child=100,
    task_acks_late=True,
    broker_transport_options={"visibility_timeout": 3600},
)