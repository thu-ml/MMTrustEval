# python test_run_george.py --task risk-identification --output_dir /data/zhangyichi/Trustworthy-MLLM/output/safety
# python test_run_george.py --task image-description --output_dir /data/zhangyichi/Trustworthy-MLLM/output/safety
# CUDA_VISIBLE_DEVICES=7 python test_run_george.py --task static_jailbreak_prompt_and_question --output_dir /data/zhangyichi/Trustworthy-MLLM/output/safety

from mmte.models import load_chatmodel, model_zoo
from mmte.perspectives import get_task, task_pool
import os
import time

# os.environ['http_proxy'] = '127.0.0.1:7890'
# os.environ['https_proxy'] = '127.0.0.1:7890'

import argparse
import os

def main(task, model_ignore, output_dir):
    # Your script logic here
    print(f"Task: {task}")
    print(f"Models to ignore: {model_ignore}")
    print(f"Output directory: {output_dir}")
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    task_handler = get_task(task)

    model_list = []
    for model_id in model_zoo():
        ignore = False
        for ig in model_ignore:
            if ig in model_id.lower():
                ignore=True
                break
        if not ignore:
            model_list.append(model_id)
    print('model list:', model_list)

    for model_id in model_list:
        if model_id != args.model_name:
            continue
        # try:
        print("="*15)
        print(f"Evaluating {model_id}...")
        if os.path.exists(os.path.join(output_dir, f"{task}_on_{model_id}.json")):
            print("="*15)
            result = task_handler.eval_response(os.path.join(output_dir, f"{task}_on_{model_id}.json"))
            print(f"Results of task {task} on {model_id}: {result}")
            print(f"Evaluating {model_id} finished.")
            print("="*15)
            continue

        for k in range(10):
            try:
                test_model = load_chatmodel(model_id)
                break
            except:
                time.sleep(3)
        
        result = task_handler.eval(test_model, os.path.join(output_dir, f"{task}_on_{model_id}.json"))
        print(f"Results of task {task} on {model_id}: {result}")
        print(f"Evaluating {model_id} finished.")
        print("="*15)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control script with command line arguments.")
    parser.add_argument("--task", type=str, required=True, help="Identifier of the task")
    parser.add_argument("--model_ignore", nargs="+", type=str, required=False, default=[], help="List of model IDs to ignore")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save logs")

    args = parser.parse_args()
    print('task_pool:', task_pool())
    
    # ['LVIS-Instruct4V', 'OpenFlamingo-3B-vitl-mpt1b', 'ShareGPT4V-7B', 'blip2_pretrain_flant5xl', 
    # 'cogvlm-chat-hf', 'gpt-3.5-turbo', 'gpt-4-1106-preview', 'gpt-4-vision-preview', 
    # 'instructblip-vicuna-7b', 'internlm-xcomposer-7b', 'kosmos2-chat', 'llama_adapter_v2', 
    # 'llava-v1.5-7b', 'lrv-instruction', 'minigpt-4-llama2-7b', 'minigpt-4-vicuna-7b-v0', 
    # 'mmicl-instructblip-t5-xxl-chat', 'mplug-owl-llama-7b', 'otter-mpt-7b-chat', 'qwen-vl-chat']
    args.model_name = 'ShareGPT4V-13B'

    for task in task_pool():
        if args.task in task:
            main(task, args.model_ignore, args.output_dir)
