import os
import pandas as pd

model_list=[
    'gpt-4-1106-vision-preview',    # 0
    'gemini-pro-vision',    # 1
    'claude-3-sonnet-20240229',    # 2
    'qwen-vl-plus',    # 3

    'minigpt-4-llama2-7b',    # 4
    'minigpt-4-vicuna-13b-v0',    # 5
    'llava-v1.5-7b',    # 6
    'llava-v1.5-13b',    # 7
    'llava-v1.6-13b',    # 8
    'ShareGPT4V-13B',    # 9
    'llava-rlhf-13b',    # 10
    'LVIS-Instruct4V',    # 11
    'otter-mpt-7b-chat',    # 12
    'internlm-xcomposer-7b',    # 13
    'internlm-xcomposer2-vl-7b',    # 14
    'mplug-owl-llama-7b',    # 15
    'mplug-owl2-llama2-7b',    # 16
    'InternVL-Chat-ViT-6B-Vicuna-13B',    # 17
    'instructblip-flan-t5-xxl',    # 18
    'qwen-vl-chat',    # 19
    'cogvlm-chat-hf',    # 20
]

# input_file='/data/zhangyichi/Trustworthy-MLLM/output/robustness/score_ood/alog.txt'
# categories=['clean','cartoon','handmake','painting','sketch','tattoo', 'weather']
# data_to_save={}
# for category in categories:
#     data_to_save[category]=[]

# with open(input_file, 'r') as f:
#     for line in f.readlines():
#         line=line.strip()
#         line=line[:-1]
#         line_list=line.split(' ')
#         model_name=line_list[4]
#         category=line_list[7]
#         acc=float(line_list[-1])

#         data_to_save[category].append(acc)


# df = pd.DataFrame(data_to_save, index = model_list)
# df.to_csv('/data/zhangyichi/Trustworthy-MLLM/output/robustness/score_ood.csv')


# input_file='/data/zhangyichi/Trustworthy-MLLM/output/robustness/score_at/alog.txt'
# categories=['clean','target','untarget']
# data_to_save={}
# for category in categories:
#     data_to_save[category]=[]

# with open(input_file, 'r') as f:
#     for line in f.readlines():
#         line=line.strip()
#         line=line[:-1]
#         line_list=line.split(' ')
#         model_name=line_list[4]
#         category=line_list[7]
#         acc=float(line_list[-1])

#         data_to_save[category].append(acc)


# df = pd.DataFrame(data_to_save, index = model_list)
# df.to_csv('/data/zhangyichi/Trustworthy-MLLM/output/robustness/score_at.csv')


input_file='/data/zhangyichi/Trustworthy-MLLM/output/robustness/score_benchlmm/alog.txt'
categories=['ct','defect','driving','game','hxray','infrard','lxray','mri','remote','robot']
data_to_save={}
for category in categories:
    data_to_save[category]=[]

with open(input_file, 'r') as f:
    for line in f.readlines():
        line=line.strip()
        line=line[:-1]
        line_list=line.split(' ')
        model_name=line_list[4]
        category=line_list[7]
        acc=float(line_list[-1])

        data_to_save[category].append(acc)


df = pd.DataFrame(data_to_save, index = model_list)
df.to_csv('/data/zhangyichi/Trustworthy-MLLM/output/robustness/score_benchlmm.csv')