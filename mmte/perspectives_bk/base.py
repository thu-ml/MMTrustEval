from abc import ABC, abstractmethod
from mmte.models import BaseChat
from typing import Optional, List, Union


class BaseEval(ABC):
    """
    Base class for evaluation tasks for different perspectives
    """
    
    EVAL_DESCRIPTION: str = ""
    task_id: str = "" # Identifier for the task
    task_list: List[str] # List of available tasks
    
    def __init__(self, task_id:str) -> None:
        self.task_id = task_id
        assert self.task_id in self.task_list, f"Task {self.task_id} is not available. Only tasks in {self.task_list} can be used."
        
    @abstractmethod
    def eval(self, test_model: BaseChat, log_response:Optional[str]=None, **kwargs):
        """
        Interface to evaluate {test_model} on a certain defined task.
        
        Arguments:
            test_model: chat model to be evaluated
            kwargs: task-specific configurations
            
        Return:
            evaluation result
        """
        raise NotImplementedError
    
    
    @abstractmethod
    def eval_response(self, log_file:Union[str|List]):
        """
        Interface to evaluate generated response on this certain task, to avoid repeated response generation
        
        Arguments:
            log_file: path to the response file
            
        Return:
            evaluation result
        """
        raise NotImplementedError
        
    @abstractmethod
    def __len__(self):
        raise NotImplementedError 
        
    def __str__(self) -> str:
        return f"""[Task Description]: {self.EVAL_DESCRIPTION} \n[Task Identifier]: {self.task_id} \n[Num of Data Points]: {len(self)} \n[Task List]: {self.task_list} """