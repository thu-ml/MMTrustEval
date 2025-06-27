import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "PKU-Alignment/Align-DS-V"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to("cpu")

processor = AutoProcessor.from_pretrained(model_id)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What is the result of this problem?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "./assets/demo.jpg" # in this repo
raw_image = Image.open(image_file)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to("cpu", torch.float16)

output = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
print(processor.decode(output[0], skip_special_tokens=True))

# <think>To solve the problem, I will first interpret the image to understand what
# mathematical operation is being represented. Then, I will perform the calculation
# based on the numbers provided in the image and confirm the result. The image shows
# a chalkboard with the equation \(18 + 23 = 41\) written on it. The numbers 18 and
# 23 are in light blue, and the result 41 is in light green. The equation \(18 + 23 = 41\)
# is presented on the chalkboard. To solve this, I will add the two numbers on the
# left side of the equation: 18 and 23. Adding these together, \(18 + 23\), I calculate
# that the sum is 41. This matches the number on the right side of the equation,
# confirming its correctness.</think>41
