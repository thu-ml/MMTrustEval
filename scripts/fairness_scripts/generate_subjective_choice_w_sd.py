# weights_path: /data/zhangyichi/Trustworthy-MLLM/playground/model_weights/stable-diffusion-xl-base-1.0
# conda_env: mllm-dev (对该环境的diffusers库进行了升级，经测试，升级可能会导致调用LAVIS的模型无法使用)
# code_demo (from https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0):

from diffusers import DiffusionPipeline
import torch
from PIL import Image
import json

pipe = DiffusionPipeline.from_pretrained("/data/zhangyichi/Trustworthy-MLLM/playground/model_weights/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

# def extract_options_from_json(json_file_path):
#     # 打开并读取 JSON 文件
#     with open(json_file_path, 'r') as file:
#         data = json.load(file)  # 解析 JSON 数据

#     # 初始化两个列表来存储选项
#     option_a_list = []
#     option_b_list = []

#     # 遍历数据集合中的每一项
#     for item in data:
#         # 提取并添加到相应的列表
#         option_a_list.append(item['option_a'])
#         option_b_list.append(item['option_b'])

    # return option_a_list, option_b_list

def extract_visual_prompts_from_file(file_path):
    visual_prompts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read all lines into a list
        lines = file.readlines()
        
        # Iterate through the lines in steps of 3, starting from the second line (index 1)
        for i in range(1, len(lines), 3):
            # Extract the line containing the visual prompt
            line = lines[i]
            # Find the start and end indices of the quotation marks
            start_quote = line.find('"')
            end_quote = line.rfind('"')
            # Extract the content within the quotation marks
            prompt = line[start_quote + 1:end_quote]
            visual_prompts.append(prompt)

    return visual_prompts

# Example usage
file_path = '/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/fairness_scripts/a_options.txt'
prompts = extract_visual_prompts_from_file(file_path)
print(len(prompts))
print(prompts)


cnt = 0
for prompt in prompts:
    cnt += 1
    images = pipe(prompt=prompt).images[0]
    images.save('/data/zhangyichi/Trustworthy-MLLM/data/fairness/subjective_choice_images/choice_a_images/{}.png'.format(cnt))
    if cnt == 120:
        break

file_path = '/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/fairness_scripts/b_options.txt'
prompts = extract_visual_prompts_from_file(file_path)
print(len(prompts))
print(prompts)


cnt = 0
for prompt in prompts:
    cnt += 1
    images = pipe(prompt=prompt).images[0]
    images.save('/data/zhangyichi/Trustworthy-MLLM/data/fairness/subjective_choice_images/choice_b_images/{}.png'.format(cnt))
    if cnt == 120:
        break