from glob import glob
import pandas as pd
import json
import os


# eval_dir = '/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug'
eval_dir = '/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5'
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

done_list = []
data = {
    'model': [],
    'task': [],
    'result': [],
}
for file in lst:
    with open(file, 'r') as fp:
        filename = os.path.basename(os.path.splitext(file)[0])
        task, model = filename.split('_on_')
        if model not in model_list:
            continue
        done_list.append(model)
        result = json.load(fp)['result']
        # print(task, model, result)
        data['model'].append(model)
        data['task'].append(task.split('-')[-1])
        data['result'].append(result)
print(data)
df = pd.DataFrame(data=data)
print(df)
df = df.pivot(index='model', columns='task', values='result')
print(df)

toname = os.path.basename(eval_dir)
df.to_excel(f'privacy_scripts/res_tab/{toname}.xlsx')
print("undone: ", set(model_list) - set(done_list))
import pdb; pdb.set_trace()