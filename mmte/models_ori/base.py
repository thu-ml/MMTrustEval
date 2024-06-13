from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


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
    



@dataclass
class Response:
    
    model_id: str
    # The identifier of the model giving the response

    content: str
    # The content of the response
    
    logprobs: Any
    # The log probabilities of the output tokens
    
    finish_reason: Optional[str]
    

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Response":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, item):
        return getattr(self, item)
    

    
    