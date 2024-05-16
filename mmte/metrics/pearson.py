from typing import Dict, Any, Sequence, List, Callable
from mmte.metrics.base import BaseDatasetMetrics, default_process
from mmte.utils.registry import registry
import numpy as np
import scipy

@registry.register_metrics()
class Pearson(BaseDatasetMetrics):
    
    metrics_id: str = 'pearson'
    metrics_ids: List[str] = ['pearson']

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def eval(self, preds: Sequence[Any], labels: Sequence[Any], process: Callable = default_process, default_value: str = "0", **kwargs) -> Dict[str, float]:
        assert len(preds) == len(labels)
        preds, labels = process(preds=preds, labels=labels)
        
        def correlation_nan_func(preds, labels):
            x_nan = np.array(preds, dtype=np.float32)
            y = np.array(labels, dtype=np.float32)
            non_nan_indices = np.where(~np.isnan(x_nan))[0]
            if non_nan_indices.size >= 2:
                correlation_nan = scipy.stats.pearsonr(x_nan[non_nan_indices], y[non_nan_indices])[0]
            else:
                correlation_nan = np.nan
            
            return correlation_nan

        def correlation_default_func(preds, labels):
            x_nan = np.array(preds, dtype=np.float32)
            x_default = np.nan_to_num(x_nan, nan=float(default_value))
            y = np.array(labels, dtype=np.float32)
            correlation_default = scipy.stats.pearsonr(x_default, y)[0]
            return correlation_default
    
        output = {
            'correlation_nan': correlation_nan_func(preds, labels),
            'correlation_default': correlation_default_func(preds, labels),
        }
        return output