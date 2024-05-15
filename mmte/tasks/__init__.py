from mmte.utils.registry import registry
from .base import BaseTask
from .confaide import ConfAIde_Task
from typing import List


def get_task(task_id) -> 'BaseTask':
    return registry.get_task_class(task_id)(task_id)

def task_pool() -> List['BaseTask']:
    return registry.list_tasks()

