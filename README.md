# Project Structure

project_root: `/data/zhangyichi/fangzhengwei/framework`

```
├── mmte
│   ├── __init__.py
│   ├── configs
│   │   ├── __init__.py
│   │   ├── datasets/*
│   │   ├── models/*
│   │   └── task/*
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── celeb.py
│   │   ├── confaide.py
│   │   ├── enron_email.py
│   │   ├── mock.py
│   │   ├── unrelatedimg.py
│   │   ├── vispr.py
│   │   └── vizwiz.py
│   ├── evaluators
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── chatmodel_eval.py
│   │   ├── classifier_eval.py
│   │   ├── metrics.py
│   │   └── rule_eval.py
│   ├── methods
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── related.py
│   │   ├── unrelated_color.py
│   │   ├── unrelated_nature.py
│   │   └── unrelated_noise.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── base.py
│   ├── tasks
│   │   └── base.py
│   └── utils
│       ├── __init__.py
│       ├── registry.py
│       └── utils.py
├── run_task.py

  
```



<p align="center">
  <img src="README.assets/image-20240522145344912.png" alt="image-20240522145344912" style="width: 40%;">
</p>

# Dataclass

source_code: `mmte/__init__.py`

dataset的输出类型有两种：

- `TxtSample`：支持纯文本的输入
  - `text`：文本prompt
  - `target`：groundtruth标签（默认为None）
  - `extra`：其他参数（默认为None）
- `ImageTxtSample`：支持多模态的输入
  - `image_path`：图像路径
  - `text`：文本prompt
  - `target`：groundtruth标签（默认为None）
  - `extra`：其他参数（默认为None）

> _OutputType = Union[ImageTxtSample, TxtSample] 

```python
@dataclass
class TxtSample:
    text: str
    target: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TxtSample":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, item):
        return getattr(self, item)
    

@dataclass
class ImageTxtSample:
    image_path: str
    text: str
    target: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageTxtSample":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, item):
        return getattr(self, item)
    

    
_OutputType = Union[ImageTxtSample, TxtSample]
```



# Models

source_code: `mmte/models/base.py`

该基类沿用之前的定义，没有任何变化

```python
class BaseChat(ABC):
    """
    Base class for models to be evaluated in a generative/chat manner.
    """
    
    model_id: str = ''   # ID for a chat model, e.g., minigpt-4-vicuna-7b-v0
    model_arch: str = '' # Architecture of the model, e.g., minigpt-4
    model_family: List[str] = [] # List of available model_ids
    
    
    def __init__(self, model_id:str) -> None:
        self.model_id = model_id
        assert self.model_id in self.model_family, f"Model {self.model_id} is not available. Only models in {self.model_family} can be used."
    
    
    @abstractmethod
    def chat(self, 
             messages: List, 
             **generation_kwargs,
             ) -> "Response":
        """
        Chat interface for generative evaluation with batch size of 1.
        
        messages: a list of messages, comprising the conversation history and following the format 
            [
                {
                    'role': 'system'/'user'/'assistant', 
                    'content': str/dict
                },
                ...
            ], 
            where content is a dict {'text': str, 'image_path': str} when it's multimodal.
        generation_kwargs: generation configuration specified for different models, including:
            temperature: float, usually between 0-2, smaller means more deterministic
            do_sample: bool, whether take sampling as the decoding strategy
            num_beams: int, the parameter for beam search
            max_new_tokens: int, maximal number of tokens to be generated
            stop_sequences: str/List[str], stop words where the model will stop generating further tokens
            output_scores: bool, whether return the logits of the generated tokens (not very practical)
        """
        raise NotImplementedError
```





# Methods

source_code: `mmte/methods/base.py`

1. 该类的主要作用是作为一个hook，作用于dataset中的getitem函数，在取样本的时候对样本进行实时的数据处理，包括但不限于对抗样本生成、相关图片生成、无关图片生成等。
2. 如果需要加载预生成的数据，可以设置lazy_mode=True，则无需重新执行run-time generation。
3. 考虑到有些数据可能只有文本信息（TxtSample），生成图片的存取路径未经定义，为了存取图片方便，可以自定义hash函数生成图片的文件名，以便后续加载预生成的数据。

```python
class BaseMethod(ABC):
    """
    Base class for methods, which can be applied to any Dataset inherits from BaseDataset as a hook in __getitem__ function.
    """

    method_id: str # Identifier for the method
    method_ids: List[str] # List of available methods
    
    def __init__(self, method_id: str, img_dir: str, lazy_mode: bool = True) -> None:
        """
        Initializing method instance.
        
        Arguments:
            method_id: Identifier for the method
            img_dir: Folder to save images
            lazy_mode: If True, it will reuse the already generated dataset. If False, it will regenerate the result
            
        Return:
            evaluation result
        """
        assert method_id in self.method_ids, f"Method {self.method_id} is not available. Only methods in {self.method_ids} can be used."
        self.method_id = method_id
        self.img_dir = img_dir
        self.lazy_mode = lazy_mode

    @abstractmethod
    def run(self, data: _OutputType, **kwargs) -> _OutputType:
        """
        Preprocess each sample in the Dataset one by one.
        
        Arguments:
            data: Union[ImageTxtSample, TxtSample], one sample in Dataset
            kwargs: extra configurations
            
            
        Return:
            processed sample
        """
        raise NotImplementedError
    
    @abstractmethod
    def hash(self, to_hash_str: str, **kwargs) -> str:
        """
        Get hash code given to_hash_str, to provide an identifier for the data sample and to reuse the generated samples.
        
        Arguments:
            to_hash_str: str
            kwargs: extra configurations
            
            
        Return:
            hash code
        """
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)
    
```





# Datasets

source_code: `mmte/datasets/base.py`

1. 该类继承自torch的dataset，可以使用torch自带的dataloader读取数据，获取到的数据类型必须是TxtSample或者ImageTxtSample类型
   - 因为dataloader不支持ImageTxtSample和TxtSample，所以在使用的时候需要用户自己实现collate_fn，默认可以使用`from mmte.datasets.base import collate_fn`
2. method_hook并不是必须的选项，当数据不需要做额外的预处理或者数据生成时，method_hook可以是None

```python
class BaseDataset(Dataset, ABC):
    """
    Base class for datasets, __getitem__ function return Union[ImageTxtSample, TxtSample].
    """

    dataset_id: str # Identifier for the dataset
    dataset_ids: Sequence[str] = [] # List of available datasets
    dataset_config: Optional[str] = "" # dataset config path

    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        """
        Initializing dataset instance.
        
        Arguments:
            dataset_id: Identifier for the dataset
            method_hook: A method instance, which is used as a preprocess hook for __getitem__ funtion
            kwargs: extra configurations
            
        """

        assert dataset_id in self.dataset_ids, f"Dataset {dataset_id} must be one of {self.dataset_ids}."
        self.dataset_id = dataset_id
        if method_hook:
            self.method_hook = method_hook
        else:
            self.method_hook = None
        self.dataset: List[Any] = []
        
    @abstractmethod
    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    @abstractmethod
    def __len__(self) -> int:
        return len(self.dataset)
```



# Evaluators

source_code: `mmte/evaluators/base.py`



## BaseEvaluator

1. 该类主要用于对chatmodel输出的结果进行评测，process的作用是对输入的preds和labels序列（可以是数值序列或文本序列）进行预处理，得到一些简单的数值序列，便于之后在eval函数中直接调用metrics函数（metrics函数只接受数值序列）。
2. 目前主要分为三大类evaluator：
   1. chatmodel evaluator：用chatmodel对结果进行处理和评估
   2. classifier evaluator：用classifier对结果打分，目前支持longformer-action-ro
   3. rule-based evaluator：包含一些拒答模版匹配和分数提取等工具
3. evaluator可以被级联起来作为evaluator sequence（具体可以参考SequentialEvaluator），主要目的是为了复用不同evaluator的process函数（例如先用chatmodel evaluator进行文本预处理，再用rule-based evaluator打分等）。

```python
class BaseEvaluator(ABC):
    """
    Base class for evaluators, to evaluate the responses from chatmodels.
    """

    evaluator_ids: List[str] = []
    def __init__(self, evaluator_id: str, metrics_cfg: Dict[str, Any]) -> None:
        """
        Initializing evaluator instance.
        
        Arguments:
            evaluator_id: Identifier for the evaluator
            metrics_cfg: config dict for metrics hooks, format: {metrics_id: metrics_kwargs, ...}
            
        """

        assert evaluator_id in self.evaluator_ids, f"Evaluator {self.evaluator_id} is not available. Only Evaluators in {self.evaluator_ids} can be used."

        self.evaluator_id = evaluator_id
        self.metrics_cfg = metrics_cfg
        for metrics_id in self.metrics_cfg.keys():
            assert metrics_id in _supported_metrics.keys(), f"{metrics_id} is not supported."

    @abstractmethod
    def process(self, preds: Sequence[Any], labels: Sequence[Any], **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        """
        1. Perform some processing on sequence data, mainly including scoring/text-extraction with chatmodel/classifier/rule-based, etc.
        2. Different evaluators can be concatenated, and the process function can be cascaded to perform multi-step processing on sequence data.
        
        Arguments:
            preds: responses from chatmodels or preds from `process` function of another evaluator
            labels: groundtruth labels or labels from `process` function of another evaluator
            
        Return:
            preds: processed preds sequence
            labels: processed labels sequence
        """

        # no-op
        return preds, labels
    
    def eval(self, preds: Sequence[Any], labels: Sequence[Any], **kwargs) -> Dict[str, Union[Sequence, float]]:
        """
        Evaluate pipeline including data processing and metrics calculation.
        
        Arguments:
            preds: responses from chatmodels
            labels: groundtruth labels
            
        Return:
            results
        """

        processed_preds, processed_labels = self.process(preds, labels)
        results = {}

        for metrics_id, kwargs in self.metrics_cfg.items():
            metrics_fn = _supported_metrics[metrics_id]
            results[metrics_id] = metrics_fn(processed_labels, processed_preds, **kwargs)
        
        return results
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.eval(*args, **kwds)
```



## SequentialEvaluator

```python
class SequentialEvaluator:
    """
    Class for cascading evaluators to perform multi-step processing on sequence data and get results from final sequence data.
    """

    def __init__(self, evaluator_seq_cfg: Dict[str, Any]) -> None:
        """
        Initializing sequence-evaluator instance.
        
        Arguments:
            evaluator_seq_cfg: config dict for instantiatizing evaluators, format: {evaluator: evaluator_kwargs, ...}
            
        """

        evaluator_seq, evaluator_cls_names = [], []
        for evaluator_id, evaluator_kwargs in evaluator_seq_cfg.items():
            evaluator_cls = registry.get_evaluator_class(evaluator_id)
            evaluator = evaluator_cls(evaluator_id, **evaluator_kwargs)
            evaluator_cls_names.append(evaluator_cls.__name__)
            evaluator_seq.append(evaluator)
        self.evaluator_seq = evaluator_seq
        self.keyname_prefix = "->".join(evaluator_cls_names)
    
    def eval(self, preds: Sequence[Any], labels: Sequence[Any], **kwargs) -> Dict[str, Union[Sequence, float]]:
        """
        Evaluate pipeline including data processing and metrics calculation.
        
        Arguments:
            preds: responses from chatmodels
            labels: groundtruth labels
            
        Return:
            results
        """
        
        for evaluator in self.evaluator_seq[:-1]:
            preds, labels = evaluator.process(preds, labels)   
        
        final_evaluator = self.evaluator_seq[-1]
        results = final_evaluator(preds, labels)
        prefix_results = {f"{self.keyname_prefix}:{key}": value for key, value in results.items()}
        return prefix_results
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.eval(*args, **kwds)
    
```



# Metrics

source_code: `mmte/evaluators/metrics.py`

1. metrics是一些简单的函数，输入是纯数值序列
2. 考虑有些evaluators经过process函数处理之后只需要对结果进行简单的聚合操作即可，因此提供了一些简单的聚合函数，如pred_no_op、pred_sum和pred_mean

```python
_supported_metrics = {
    # aggregation op
    "pred_no_op": pred_no_op,
    "pred_sum": pred_sum,
    "pred_mean": pred_mean,

    # general metrics
    "accuracy_score": accuracy_score,
    "precision_score": precision_score,
    "recall_score": recall_score, 
    "f1_score": f1_score,
    "pearson_corr": pearson_corr,
    "failure": failure,
}

```





# Tasks

## BaseTask

source_code: `mmte/tasks/base.py`

1. 支持从run_task.py入口one-cmd执行评测任务，无需继承BaseTask
2. 通过从 `task_config.yaml`读取配置或者通过添加 `cfg-options`命令行参数对配置进行覆盖或者添加



## task_config

- model_id：模型的id

- dataset_id：数据集的id

- log_file：结果输出的json文件路径

- method_cfg：method的config，{method_id: method_kwargs, ...}，default：{}

- dataset_cfg：dataset_kwargs，初始化dataset时需要的额外参数，default：{}

- evaluator_seq_cfgs：evaluator sequence的config，List中表示不同的evaluator sequence，一个sequence里面有多个evaluator，一个evaluator对应多个metrics，default：[]

  ```markdown
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

  



# Tutorial to add Tasks

以privacy的confaide_text（+unrelated-color-image）为例：

1. 定义dataset，并注册dataset_id：confaide-text（可参考`mmte/datasets/confaide.py`）

2. 定义evaluator，并注册evaluator_id：rule_match_and_score_eval（可参考`mmte/evaluators/rule_eval.py`）

   - 如果需要定义额外的metrics需要自行修改`mmte/evaluators/metrics.py`并在`_supported_metrics`上注册metrics_fn

3. 编辑task_config.yaml（可参考`mmte/configs/task/confaide_text_color.yaml`）

   ```yaml
   dataset_id: "confaide-text"
   model_id: "llava-v1.5-7b"
   
   method_cfg:
     {
       unrelated-image-color:
         {
           "img_dir": "./logs/privacy/confaide-text-color",
           "img_size": [50, 50],
           "lazy_mode": True,
         },
     }
   log_file: "./logs/privacy/confaide-text-color.json"
   
   evaluator_seq_cfgs:
     [
       {
         "rule_match_and_score_eval":
           { metrics_cfg: { pearson_corr: {}, failure: {} } },
       },
     ]
   
   ```

4. 一键运行：

   ```shell
   python run_task.py --config mmte/configs/task/confaide_text_color.yaml
   ```

