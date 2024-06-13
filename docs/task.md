
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

This is to be done.

- model_id：模型的id

- dataset_id：数据集的id

- log_file：结果输出的json文件路径

- method_cfg：method的config，{method_id: method_kwargs, ...}，default：{}

- dataset_cfg：dataset_kwargs，初始化dataset时需要的额外参数，default：{}

- evaluator_seq_cfgs：evaluator sequence的config，List中表示不同的evaluator sequence，一个sequence里面有多个evaluator，一个evaluator对应多个metrics，default：[]



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



