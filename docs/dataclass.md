
# Dataclass

We primarily define two dataclasses to contain the multimodal data to be processed by MLLMs, one for text-only samples and the other for image-text pairs. The detailed attributes in the dataclass are introduced below. 

- `TxtSample`: to support text-only sample
    - `text`: prompt in text
    - `target`: ground-truth label（Default: None）
    - `extra`: auxiliary arguments that may help in the process afterwards, e.g., adversarial example generation（Default: None）


- `ImageTxtSample`: to support multimodal input, i.e., an image-text pair
    - `image_path`: path to the image file
    - `text`: prompt in text
    - `target`: ground-truth label（Default: None）
    - `extra`：auxiliary arguments that may help in the process afterwards, e.g., adversarial example generation（Default: None）

The type of the output from an MLLM is also restricted to these two dataclasses.

- `_OutputType = Union[ImageTxtSample, TxtSample]`


Source code in `mmte/__init__.py`.
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


