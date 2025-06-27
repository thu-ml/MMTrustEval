from PIL import Image

from transformers import AutoTokenizer, AutoProcessor
from decord import VideoReader, cpu    # pip install decord

import torch
from transformers import AutoConfig, AutoModel
model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
# config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# model = mPLUGOwl3Model(config).cuda().half()
model = AutoModel.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half, trust_remote_code=True)
model.eval().cuda()


model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)

image = Image.new('RGB', (500, 500), color='red')

messages = [
#     {"role": "user", "content": """<|image|>
# Describe this image."""},
    {"role": "user", "content": """tell me a joke."""},
    {"role": "assistant", "content": ""}
]

# inputs = processor(messages, images=[image], videos=None)
inputs = processor(messages, images=None, videos=None)

inputs.to('cuda')
inputs.update({
    'tokenizer': tokenizer,
    'max_new_tokens':100,
    'decode_text':True,
})


g = model.generate(**inputs)
print(g)
