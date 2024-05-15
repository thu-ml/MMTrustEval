import os

model_ids=[ 
    "ShareGPT4V-7B",
    "cogvlm-chat-hf",
    "internlm-xcomposer-7b",
    "llava-v1.5-7b",
    "minigpt-4-vicuna-13b-v0",
    "otter-mpt-7b-chat",
    "qwen-vl-chat",
    "mplug-owl2-llama2-7b",
    "minigpt-4-llama2-7b",
    # "gpt-4-1106-vision-preview",
]

task_ids=[
    "pii-query-email-name-occupation",
    "pii-query-email-wo-name-occupation",
    "pii-query-phone-name-occupation",
    "pii-query-phone-wo-name-occupation",
    "pii-query-address-name-occupation",
    "pii-query-address-wo-name-occupation",
    "pii-query-email-name-wo-occupation",
    "pii-query-email-wo-name-wo-occupation",
    "pii-query-phone-name-wo-occupation",
    "pii-query-phone-wo-name-wo-occupation",
    "pii-query-address-name-wo-occupation",
    "pii-query-address-wo-name-wo-occupation",
]

dirname = '/data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug'

name_template = '{task}_on_{model_id}.json'

for model_id in model_ids:
    missing = []
    for task in task_ids:
        filename = os.path.join(dirname, name_template.format(task=task, model_id=model_id))
        if not os.path.exists(filename):
            missing.append(task)
    if missing:
        print((f'{model_id} missing tasks:\n{missing}\n\n'))

