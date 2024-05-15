from mmte.utils.registry import registry
from mmte.perspectives.base import BaseEval
from mmte.perspectives.privacy import *
from mmte.perspectives.safety import *
from mmte.perspectives.fairness import *
from mmte.perspectives.truthfulness import *
# from mmte.perspectives.robustness import *
from typing import List


def get_task(task_id) -> 'BaseEval':
    return registry.get_task_class(task_id)(task_id)

def task_pool() -> List['BaseEval']:
    return registry.list_tasks()

