from typing import Dict, Any, Sequence, List, Callable
from mmte.metrics.base import BasePerSampleMetrics, default_process
from mmte.utils.registry import registry

@registry.register_metrics()
class Accuracy(BasePerSampleMetrics):
    
    metrics_id: str = 'accuracy'
    metrics_ids: List[str] = ['accuracy']
    
    def __init__(self) -> None:
        super().__init__()

    def eval(self, preds: Sequence[Any], labels: Sequence[Any], process: Callable = default_process, **kwargs) -> Dict[str, Sequence]:
        acc = []
        preds, labels = process(preds=preds, labels=labels)
        for pred, label in zip(preds, labels):
            assert isinstance(pred, str) and isinstance(label, str)
            if label in pred:
                acc.append(1)
            else:
                acc.append(0)
        
        output = {
            "accuracy": acc,
        }
        return output
    