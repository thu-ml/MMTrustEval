from abc import ABC, abstractmethod
from typing import Optional, List, Union, Any, Sequence, Tuple, Dict
from torch.utils.data import DataLoader
from mmte.datasets.base import BaseDataset
from mmte.metrics.base import BaseDatasetMetrics, BasePerSampleMetrics
from mmte.utils.registry import registry
from mmte.methods.base import BaseMethod
from mmte.models.base import BaseChat
import json

def get_task(task_id) -> 'BaseTask':
    return registry.get_task_class(task_id)(task_id)

def task_pool() -> List['BaseTask']:
    return registry.list_tasks()

class BaseTask(ABC):    
    def __init__(self, dataset_id: str, model_id: str, method_id: Optional[str], metrics_ids: List[str]) -> None:
        self.dataset_id = dataset_id
        self.metrics_ids = metrics_ids
        self.method_id = method_id
        self.model_id = model_id
    
    def get_handlers(self) -> None:
        self.metrics_list = self.get_metrics()
        self.dataset = self.get_dataset()
        self.method = self.get_method()
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
        method_cls = registry.get_method_class(self.method_id)
        method = method_cls(self.method_id)
        return method

    def get_dataset(self) -> BaseDataset:
        dataset_cls = registry.get_dataset_class(self.dataset_id)
        dataset = dataset_cls(self.dataset_id)
        if self.method_id and self.method:
            poisoned_dataset = self.method(dataset)
            return poisoned_dataset
        return dataset
    
    @abstractmethod
    def get_dataloader(self) -> DataLoader:
        raise NotImplementedError
    
    @abstractmethod
    def pre_processing(self, preds: Sequence[Any], labels: Sequence[Any], **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        # return preds, labels
        raise NotImplementedError
    
    @abstractmethod
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