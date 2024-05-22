from abc import abstractmethod, ABC
from typing import Any, List
from mmte import _OutputType

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
    