from typing import Dict, Any, Sequence, List
from mmte.metrics.base import BasePerSampleMetrics
from mmte.utils.registry import registry

@registry.register_metrics()
class Accuracy(BasePerSampleMetrics):
    
    metrics_id: str = 'accuracy'
    metrics_ids: List[str] = ['accuracy']
    
    def __init__(self) -> None:
        super().__init__()

    def eval(self, preds: Sequence[Any], labels: Sequence[Any]) -> Dict[str, Sequence]:
        acc = []
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
    