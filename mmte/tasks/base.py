from abc import ABC, abstractmethod
from typing import Optional, List, Union, Any, Type
from torch.utils.data import Dataset, DataLoader
from mmte.datasets.base import BaseDataset
from mmte.metrics.base import BaseDatasetMetrics, BasePerSampleMetrics
from mmte.utils.registry import registry
from mmte.methods.base import BaseMethod
from mmte.processes.base import BaseProcess
from mmte.models.base import BaseChat
import json
import os


class BaseTask(ABC):
    
    TASK_DESCRIPTION: str = ""
    task_id: str = "" # Identifier for the task
    task_ids: List[str] = []
    
    supported_model_list: List[str]
    supported_method_list: List[str]
    supported_dataset_list: List[str] 
    supported_metrics_list: List[str]
    # supported_process_list: List[str]

    # def __init__(self, task_id: str, dataset_id: str, model_id: str, method_id: Optional[str], process_id: Optional[str], metrics_id: str) -> None:
    def __init__(self, task_id: str, dataset_id: str, model_id: str, method_id: Optional[str], metrics_id: str) -> None:
        assert dataset_id in self.supported_dataset_list, f"Dataset {dataset_id} is not available. Only datasets in {self.supported_dataset_list} can be used."
        assert metrics_id in self.supported_metrics_list, f"Metrics {metrics_id} is not available. Only metrics in {self.supported_metrics_list} can be used."
        assert model_id in self.supported_model_list, f"Model {model_id} is not available. Only models in {self.supported_model_list} can be used."
        assert method_id is None or \
            method_id in self.supported_method_list, f"Method {self.method_id} is not available. Only methods in {self.supported_method_list} can be used."
        # assert process_id is None or \
        #     process_id in self.supported_process_list, f"Process {self.process_id} is not available. Only processes in {self.supported_process_list} can be used."
        
        self.task_id = task_id
        self.dataset_id = dataset_id
        self.metrics_id = metrics_id
        self.method_id = method_id
        self.model_id = model_id
        # self.process_id = process_id
        self.get_handlers()
    
    def get_handlers(self) -> None:
        self.metrics = self.get_metrics()
        self.dataset = self.get_dataset()
        self.method = self.get_method()
        self.model = self.get_model()
        # self.process = self.get_process()

    # TODO: multi-metrics
    def get_metrics(self) -> List[Union[BaseDatasetMetrics, BasePerSampleMetrics]]:
        metrics_cls = registry.get_metrics_class(self.metrics_id)
        metrics = metrics_cls()
        return metrics
    
    # def get_process(self) -> BaseProcess:
    #     process_cls = registry.get_process_class(self.process_id)
    #     process = process_cls(self.process_id)
    #     return process
    
    # TODO: multi-
    def get_model(self) -> BaseChat:
        model_cls = registry.get_chatmodel_class(self.model_id)
        model = model_cls(self.model_id)
        return model
    
    # TODO: multi- [images/texts augmentation]
    # TODO: method - surrogate model [not target model]
    def get_method(self) -> BaseMethod:
        if self.method_id is None:
            return None
        method_cls = registry.get_method_class(self.method_id)
        method = method_cls(self.method_id)
        return method

    # TODO: multi-datasets List[]
    def get_dataset(self) -> BaseDataset:
        dataset_cls = registry.get_dataset_class(self.dataset_id)
        dataset = dataset_cls(self.dataset_id)
        if self.method_id and self.method:
            poisoned_dataset = self.method(dataset)
            return poisoned_dataset

        return dataset

    def run_task_from_scratch(self, log_file:Optional[str]=None, **kwargs):
        dataloader = DataLoader(dataset=self.dataset)
        
        responses = []
        for data in dataloader:
            image_path = data['image_path']
            text = data['text']
            target = data['target']

            message = [
                        {
                            "role": "user",
                            "content": {"text": text, \
                                        "image_path": image_path}
                        }
                    ]
            response = self.model.chat(messages=message, max_new_tokens=50, do_sample=False)
            responses.append({"image": image_path, "prompt": text, "response": response.content, "label": target})

        log = {}
        log['raw_log'] = responses

        preds = [response['response'] for response in responses]
        labels = [response['label'] for response in responses]
        
        if isinstance(self.metrics, BaseDatasetMetrics):
            results = self.metrics(preds, labels, process=self.process)
        elif isinstance(self.metrics, BasePerSampleMetrics):
            results = self.metrics(preds, labels)
            # TODO: sequence to scalar 
        else:
            raise NotImplementedError


        log['result'] = results
        
        if log_file is not None:
            with open(log_file, "w") as f:
                json.dump(log, f, indent=4)
        

    def run_eval_from_json(self, log_file: str, **kwargs):
        
        with open(log_file, "r") as f:
            log = json.load(f)

        responses = log['raw_log']
        preds = [response['response'] for response in responses]
        labels = [response['label'] for response in responses]
        
        if isinstance(self.metrics, BaseDatasetMetrics):
            results = self.metrics(preds, labels, process=self.process)
        elif isinstance(self.metrics, BasePerSampleMetrics):
            results = self.metrics(preds, labels)
        else:
            raise NotImplementedError

        log['result'] = results
        
        if log_file is not None:
            with open(log_file, "w") as f:
                json.dump(log, f, indent=4)
        

    
    