from glob import glob
import pandas as pd
import json
import os
from collections import defaultdict


eval_dir = '/data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug'
lst = glob(f'{eval_dir}/*.json')
model_list = [
    "minigpt-4-llama2-7b",
    "minigpt-4-vicuna-13b-v0",
    "llava-v1.5-7b",
    "llava-v1.5-13b",
    "ShareGPT4V-13B",
    "llava-rlhf-13b",
    "LVIS-Instruct4V",
    "otter-mpt-7b-chat",
    "internlm-xcomposer-7b",
    "internlm-xcomposer2-vl-7b",
    "mplug-owl-llama-7b",
    "mplug-owl2-llama2-7b",
    "InternVL-Chat-ViT-6B-Vicuna-13B",
    "qwen-vl-chat",
    "cogvlm-chat-hf",
    "llava-v1.6-13b",
    "instructblip-flan-t5-xxl",
    "gpt-4-1106-vision-preview",
    "gemini-pro-vision",
    "claude-3-sonnet-20240229",
    "qwen-vl-plus",
]

results_name = [
    "leakage_rate",
    "refusal_rate",
]
data = defaultdict(list)
for file in lst:
    with open(file, 'r') as fp:
        filename = os.path.basename(os.path.splitext(file)[0])
        task, model = filename.split('_on_')
        if model not in model_list:
            continue
        results = json.load(fp)['result']
        
        data['task'].append(task.split('pii-query-')[-1])
        data['model'].append(model)
        data['leakage_rate'].append(results['leakage_rate'])
        data['refusal_rate'].append(results['refusal_rate'])
            
print(data)
df = pd.DataFrame(data=data)
print(df)
df = df.pivot_table(index=['model', 'task'])

toname = os.path.basename(eval_dir)
df.to_excel(f'privacy_scripts/res_tab/{toname}.xlsx')
import pdb; pdb.set_trace()