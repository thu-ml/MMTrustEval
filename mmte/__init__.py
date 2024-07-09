import os
import yaml
from pprint import pformat
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Union

lib_path = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.join(lib_path, '..')


APIKEY_FILE = 'env/apikey.yml'
apikey_file = os.getenv("APIKEY_FILE", APIKEY_FILE)
with open(apikey_file, 'r') as f:
    apikeys = yaml.load(f, Loader=yaml.FullLoader)
os.environ.update(apikeys)
if apikeys:
    print("apikeys loaded: \n{}".format(pformat(apikeys, indent=2)))


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