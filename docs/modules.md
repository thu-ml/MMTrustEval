
# Modules

In this section, we introduce each element in the [task flow](#flow) about their interface, typical usage, etc., in the order of their parts in the workflow. All elements are registered by an unique identifier into the global `registry` (defined in `mmte/utils/registry.py`) and can be accessed by the `registry.get_**_class(id)` method.




##  Datasets

Datasets are defined to collect the samples to be tested for a specific task. It provides the prompt, image path, labels and possibly other information about the data point to the following process. Here are some technical notes about this class.

1. The class is a subclass of the `torch.utils.data.Dataset` and users can iterate through the dataset by the default `torch.utils.data.Dataloader`. To customize a dataset, user need to define `__getitem__` and `__len__` as usual along with a `collate_fn` so that the dataloader can support the dataclass of `TxtSample` and `ImageTxtSample`. We provide a default one in `mmte.datasets.base.collate_fn` which should work for most cases. <span style="color:blue">(Yichi: Support customized collate_fn ?)</span>
2. A method to further process the data for a certain task, which can be independent from the original dataset, can be optionally specified via the argument `method_hook` when initialization. This could make additional augmentation, attack and other preprocessing to the existing datasets more convenient. This is illustrated in the [Method](#method) part.
3. Some information about the dataset can be configured through a yaml config file, like the directory of images, the path to the annotation file. 
4. `dataset_ids` is the list of supported `dataset_id` for this class, which specify the different splits and processors in sample preparing.


<!-- 1. 该类继承自torch的dataset，可以使用torch自带的dataloader读取数据，获取到的数据类型必须是TxtSample或者ImageTxtSample类型
   - 因为dataloader不支持ImageTxtSample和TxtSample，所以在使用的时候需要用户自己实现collate_fn，默认可以使用`from mmte.datasets.base import collate_fn`
2. method_hook并不是必须的选项，当数据不需要做额外的预处理或者数据生成时，method_hook可以是None -->


Source code in `mmte/datasets/base.py`. Refer to `mmte/datasets/celeb.py` for an example.
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




## <a name="method"></a> Methods

Methods are designed for additional data processing independent from and universal across datasets, for instance, generating adversarial examples, pairing text prompts with diverse images. Users do not need to modify the code for datasets but only implement a new class of `Method` and pass it as a hook to the dataset. Here are some technical notes about this class.

1. This class works as a hook in the function `__getitem__` of a dataset, which is optional. 
2. For cases where new images are generated with time-consuming methods or reproducibility is needed, we can set the `lazy_mode=True` to utilize the previously generated samples. To tackle the challenge that text-only data may not have clear identifiers pointing to a sample, a `hash` function can be defined to generate the filename for the generated data.

<!-- 1. 该类的主要作用是作为一个hook，作用于dataset中的getitem函数，在取样本的时候对样本进行实时的数据处理，包括但不限于对抗样本生成、相关图片生成、无关图片生成等。
2. 如果需要加载预生成的数据，可以设置lazy_mode=True，则无需重新执行run-time generation。
3. 考虑到有些数据可能只有文本信息（TxtSample），生成图片的存取路径未经定义，为了存取图片方便，可以自定义hash函数生成图片的文件名，以便后续加载预生成的数据。 -->

Source code in `mmte/methods/base.py`. Refer to `mmte/methods/unrelated_color.py` for an example.
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



##  Models

Models encapsulate the chat model of MLLMs into a unified interface for inference. This enables more convenient standardized evaluation of diverse models. Here are some technical notes about this class.

1. `chat` unifies the interface for generation. `messages` is a list representing the conversation history, `generation_kwargs` is the generation configuration, indicating whether to `do_sample`, the setting of `temperature`, `max_new_tokens`, etc. The setting of generation configuration follows that in huggingface transformers. <span style="color:blue">(Yichi: Generation kwargs should be able to be customized in config file)</span>
2. `model_id` is the unique identifier to get the chatmodel from `registry_getchatmodel_class` and `model_family` defines the list of available model identifiers.




Source code in `mmte/models/base.py`. Refer to `mmte/models/openai_chat.py` for an example.
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









##  Evaluators

source_code: `mmte/evaluators/base.py`



### BaseEvaluator

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



### SequentialEvaluator

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



##  Metrics


We pre-define some common metrics for users to call. These functions to calculate metrics take two array-like arguments of digits to compute the statistical or sample-wise results. We also consider some cases where simple operations of aggregation are needed, e.g., sum, mean.


Source code in `mmte/evaluators/metrics.py`.

```python
"""
Input Requirement
y_true: 1d array-like
y_pred: 1d array-like
"""

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





##  Tasks

### BaseTask

source_code: `mmte/tasks/base.py`

1. 支持从run_task.py入口one-cmd执行评测任务，无需继承BaseTask
2. 通过从 `task_config.yaml`读取配置或者通过添加 `cfg-options`命令行参数对配置进行覆盖或者添加



### task_config

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




