"""Celery application for Quantoshi background tasks.

Tasks:
  - run_mc_simulation: MC price paths + Citadel/Retire/DCA simulation
  - fetch_btc_price: periodic price ticker update
  - fetch_sparkline: periodic 24h sparkline update

Usage:
  celery -A btc_web.celery_app worker --loglevel=info -c 2
  celery -A btc_web.celery_app beat --loglevel=info
"""
from celery import Celery

celery_app = Celery(
    'quantoshi',
    broker='redis://localhost:6379/1',
    backend='redis://localhost:6379/2',
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_soft_time_limit=60,      # warn after 60s
    task_time_limit=120,          # kill after 120s
    worker_max_tasks_per_child=50, # recycle workers to prevent leaks
    result_expires=3600,          # task results expire after 1 hour
    task_acks_late=True,          # re-queue if worker dies mid-task
    worker_prefetch_multiplier=1, # don't prefetch — tasks are long-running
)

# Periodic tasks (Celery Beat)
celery_app.conf.beat_schedule = {
    'fetch-btc-price': {
        'task': 'btc_web.tasks.fetch_btc_price',
        'schedule': 1200.0,  # every 20 minutes (matches frontend interval)
    },
    'fetch-sparkline': {
        'task': 'btc_web.tasks.fetch_sparkline',
        'schedule': 300.0,   # every 5 minutes
    },
}
