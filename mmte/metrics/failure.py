from typing import Dict, Any, Sequence, List, Callable
from mmte.metrics.base import BaseDatasetMetrics, default_process
from mmte.utils.registry import registry
import numpy as np

@registry.register_metrics()
class Failure(BaseDatasetMetrics):
    
    metrics_id: str = 'failure'
    metrics_ids: List[str] = ['failure']

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def eval(self, preds: Sequence[Any], labels: Sequence[Any], process: Callable = default_process, **kwargs) -> Dict[str, float]:
        assert len(preds) == len(labels)
        preds, labels = process(preds=preds, labels=labels)

        def failure_rate_func(preds):
            x_nan = np.array(preds, dtype=np.float32)
            failure = np.isnan(x_nan).sum() / x_nan.size
            return failure
        

        output = {
            'failure': failure_rate_func(preds),
        }
        return output