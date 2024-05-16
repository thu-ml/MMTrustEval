from typing import List, Union, Sequence, Any, Dict, Tuple
from mmte.tasks.base import BaseTask
from mmte.utils.registry import registry
from mmte.processes import Txt2Score
import numpy as np

class ConfAIde_Task(BaseTask):
    def pre_processing(self, preds: Sequence[Any], labels: Sequence[Any], **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        process_id = 'txt2score'
        process_cls = registry.get_process_class(process_id)
        process: Txt2Score = process_cls(process_id)
        preds = process.process(preds, labels, fail_id=np.nan)
        return preds, labels
    
    def post_processing(self, responses: List[Dict[str, Any]]) -> Dict[str, Union[float, Sequence]]:
        preds: Sequence[str] = [response['response'] for response in responses]
        labels: Sequence[str] = [response['target'] for response in responses]

        results = {}
        
        for metric_id, metric in self.metrics_list.items():
            print(f"start running evaluation on metrics: {metric_id} ...")
            result: Dict = metric(preds, labels, process=self.pre_processing)
            # user custom post-processing needed here
            # For example, process the sequential results from outputs of BasePerSampleMetrics

            results.update(result)
        
        return results