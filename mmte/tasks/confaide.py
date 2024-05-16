from typing import Optional, List, Union, Sequence, Any, Dict, Tuple, Type
from torch.utils.data import DataLoader
from mmte.datasets.base import BaseDataset, collate_fn
from mmte.metrics.base import BaseDatasetMetrics, BasePerSampleMetrics
from mmte.methods.base import BaseMethod
from mmte.models.base import BaseChat
from mmte.tasks.base import BaseTask
from mmte.utils.registry import registry
from mmte.processes import Txt2Score
import numpy as np
import json

class ConfAIde_Task(BaseTask):
    supported_model_list: List[str] = ['llava-v1.5-7b', 'minigpt-4-llama2-7b']
    supported_method_list: Optional[List[str]] = ['unrelated-image-color', 'unrelated-image-nature', 'unrelated-image-noise'] # methods for text dataset
    supported_dataset_list: List[str] = ["confaide-text", "confaide-image", "confaide-unrelated-image-color", "confaide-unrelated-image-nature", "confaide-unrelated-image-noise"]
    supported_metrics_list: List[str] = ['pearson', 'failure']
    
    def __init__(self, task_id: str, dataset_id: str, model_id: str, metrics_ids: List[str], method_id: Optional[str] = None, cfg: Optional[Dict] = None) -> None:
        
        self.dataset_id = dataset_id
        self.metrics_ids = metrics_ids
        self.method_id = method_id
        self.model_id = model_id
        self.task_id = task_id
        self.cfg = cfg
    
    def get_handlers(self) -> None:
        self.metrics_list = self.get_metrics()
        self.method = self.get_method() # get method before dataset
        self.dataset = self.get_dataset()
        self.model = self.get_model()

    def get_metrics(self) -> List[Union[BaseDatasetMetrics, BasePerSampleMetrics]]:
        metrics_list = {}
        for metrics_id in self.metrics_ids:
            metrics_cls = registry.get_metrics_class(metrics_id)
            metrics_list[metrics_id] = metrics_cls()
        return metrics_list
    
    def get_model(self) -> BaseChat:
        model_cls = registry.get_chatmodel_class(self.model_id)
        model = model_cls(self.model_id)
        return model
    
    def get_method(self) -> BaseMethod:
        if self.method_id is None:
            return None
        
        if self.cfg:
            method_cfg = self.cfg.get('method_cfg', {})
        else:
            method_cfg = {}

        method_cls = registry.get_method_class(self.method_id)
        method = method_cls(self.method_id, **method_cfg)
        return method

    def get_dataset(self) -> BaseDataset:
        dataset_cls: Type[BaseDataset] = registry.get_dataset_class(self.dataset_id)
        dataset = dataset_cls(self.dataset_id, method_hook=self.method)
        return dataset
    
    def get_dataloader(self) -> DataLoader:
        dataloader = DataLoader(dataset=self.dataset, batch_size=1, collate_fn=collate_fn)
        return dataloader
    
    def pre_processing(self, preds: Sequence[Any], labels: Sequence[Any], fail_id: Union[float, np.ndarray] = np.nan, **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        process_id = 'txt2score'
        process_cls = registry.get_process_class(process_id)
        process: Txt2Score = process_cls(process_id)
        preds = process.process(preds, labels, fail_id)
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

    def save_results(self, results: Dict[str, Any], log_file: Optional[str] = None) -> None:
        if log_file is not None:
            with open(log_file, "w") as f:
                json.dump(results, f, indent=4)

    def generate(self, dataloader: DataLoader, **generate_kwargs) -> List[Dict[str, Any]]:
        print('len(self.dataset): ', len(dataloader.dataset))
        responses = []
        for batch_data in dataloader:
            for data in batch_data:
                """
                    # for text data
                    message = [
                        {
                            "role": "user",
                            "content": text
                        }
                    ]

                    # for multimodal data
                    message = [
                        {
                            "role": "user",
                            "content": {
                                "image_path": ...,
                                "text": ...
                            }
                        }
                    ]
                """
                
                message = data['message']
                target = data['target']
                extra: Dict[str, Any] = data['extra']

                response = self.model.chat(messages=message, max_new_tokens=50, do_sample=False, **generate_kwargs)
                output = {
                    "content": message[0]['content'],
                    "response": response.content,
                    "target": target,
                    "extra": extra,
                }
            
                responses.append(output)
        
        return responses
        
    def pipeline(self) -> None:
        self.get_handlers()
        dataloader = self.get_dataloader()

        responses = self.generate(dataloader)
        results = self.post_processing(responses)
        self.save_results(results, log_file='./out.json')