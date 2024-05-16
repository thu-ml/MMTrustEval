import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

lib_path = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.join(lib_path, '..')

@dataclass
class TxtSample:
    text: str
    target: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    
@dataclass
class ImageTxtSample:
    image_path: str
    text: str
    target: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


_OutputType = Union[ImageTxtSample, TxtSample]