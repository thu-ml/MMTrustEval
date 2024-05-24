
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


