from abc import abstractmethod, ABC
from typing import Any, List
from mmte import _OutputType

class BaseMethod(ABC):

    method_id: str
    method_ids: List[str]
    
    def __init__(self, method_id: str) -> None:
        assert method_id in self.method_ids
        self.method_id = method_id

    @abstractmethod
    def run(self, data: _OutputType, **kwargs) -> _OutputType:
        raise NotImplementedError
    
    @abstractmethod
    def hash(self, to_hash_str: str, **kwargs) -> str:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)
    