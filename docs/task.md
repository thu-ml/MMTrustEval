
#  Task


A task powered by MMTrustEval is a collection of the aforementioned modules, i.e., a certain model is evaluated on a certain dataset with certain metrics. Here, we introduce the basic logic of a task workflow in MMTrustEval and how to construct a task via a configuration file. 

Note that, the purpose of MMTrustEval is to provide modularized tools for developing new tasks to evaluate trustworthiness of MLLMs, instead of to restrict the implementation to modularized configuration. While the tasks in MultiTrust are organized with these modules, users are free to implement their own tasks in a more customized style and only use one or several modules like unified models and off-the-shelf evaluators to facilitate their evaluation.


## Task Pipeline

We provide a `BaseTask` class to organize the modules in a standardized logic and provide a way for modular customization. The modules for different elements are instantiated according to the configuration from a yaml file or commandline argurments automatically and the user can simply launch the evaluation with one command. 

Source code in `mmte/tasks/base.py`

```python
class BaseTask(ABC):    
    def __init__(self, dataset_id: str, model_id: str, method_cfg: Optional[Dict] = {}, 
                dataset_cfg: Optional[Dict] = {}, evaluator_seq_cfgs: List = [], 
                log_file: Optional[str] = None) -> None:
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.method_cfg = method_cfg
        self.dataset_cfg = dataset_cfg
        self.evaluator_seq_cfgs = evaluator_seq_cfgs
        self.log_file = log_file

    def pipeline(self) -> None:
        self.get_handlers() # automatic module instantiation
        dataloader = self.get_dataloader()
        responses = self.generate(dataloader) # unified model inference
        results = self.eval(responses) # response postprocessing and metrics computation
        self.save_results(results) # result logging
```



## Task Configuration

- `model_id`: The ID of the model

- `dataset_id`: The ID of the dataset

- `log_file`: The file path for the output JSON file

- `method_cfg`: Configuration for the method, formatted as `{method_id: method_kwargs, ...}`. Default: `{}`

- `dataset_cfg`: Additional parameters required for initializing the dataset, formatted as `dataset_kwargs`. Default: `{}`

- `generation_kwargs`: Additional parameters used for inference with the chat model. Default: `{}`

- `evaluator_seq_cfgs`: Configuration for the evaluator sequence. Each list item represents a different evaluator sequence, with each sequence containing multiple evaluators, and each evaluator corresponding to multiple metrics. Default: `[]`



```python
List[
    {evaluator_id: 

        {metrics_cfg: 

            {metrics_id: metrics_kwargs},
            {metrics_id: metrics_kwargs},
            ...
            
        },
        ...
    },
    ...
]
```



