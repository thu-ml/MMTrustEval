from abc import abstractmethod, ABC
from typing import Any, Sequence, Union, List
import numpy as np

class BaseProcess(ABC):

    process_id: str
    process_ids: List[str]
    
    def __init__(self, process_id: str) -> None:
        self.process_id = process_id

    @abstractmethod
    def process(self, preds: Sequence[Any], labels: Sequence[Any], fail_id: Union[float, np.ndarray]) -> Sequence[Union[float, np.ndarray]]:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.process(*args, **kwds)
    