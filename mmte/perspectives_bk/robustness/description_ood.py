import sys
sys.path.append('/data/zhangyichi/Trustworthy-MLLM/MMTrustEval')

import os
import json
import argparse
import mmte
from mmte.models import load_chatmodel, model_zoo
import time

class OODEval():
    def __init__(self, args, model, model_name, root_dir='', out_dir='', log_file='') -> None:
        self.args = args
        self.model = model
        self.model_name = model_name
        self.root_dir = root_dir
        self.out_dir = out_dir
        self.log_file = log_file
        os.makedirs(self.out_dir, exist_ok=True)
        self.categories = {
            'clean': 'coco2017/val2017/',
            'cartoon': 'coco-o/cartoon/val2017/',
            'handmake': 'coco-o/handmake/val2017/',
            'painting': 'coco-o/painting/val2017/',
            'sketch': 'coco-o/sketch/val2017/',
            'tattoo': 'coco-o/tattoo/val2017/',
            'weather': 'coco-o/weather/val2017/'
        }
        # self.label_dirs = ['coco-labels-sample-100',]
    
    def eval(self, **generation_kwargs):
        print(f'Processing model {self.model_name}.')
        start_time = time.time()
        with open(self.log_file, 'w') as log_output:
            for category, image_dir in self.categories.items():
                start_time_each=time.time()
                with open(os.path.join(self.root_dir, 'coco-labels-sample-100', f'{category}.json'), 'r') as f:
                    coco_dict=json.load(f)
                
                # evaluation
                # correct_num=0
                # total_num=0
                response_list=[]
                for image_id, image_info in coco_dict.items():
                    try:
                        image_name=image_info['image_name']
                        label=image_info['label']
                        text=f'Please provide a detailed description of the image.'

                        image_path=os.path.join(self.root_dir, image_dir, image_name)
                        
                        messages = [{
                            "role": "user",
                            "content": {
                                "text": text,
                                "image_path": image_path
                                },
                            }]
                        response = self.model.chat(messages, **generation_kwargs)
                        response_txt = response.content if isinstance(response.content, str) else response.content[0]
                        # response_txt.replace('\t', ' ').replace('\n', ' ').replace('  ', ' ')
                    except:
                        log_output.write(f'Get error in processing image {image_id} of category {category}.')
                        response_txt=None
                    response_list.append({'image_id': image_id, 'image_name': image_name, 'question': text, 'gt_caption': None, 'gt_label': label, 'response': response_txt})
                    
                with open(os.path.join(self.out_dir, f'{category}_response.json'), 'w') as out_file:
                    for response_each in response_list:
                        out_line=json.dumps(response_each)
                        out_file.write(out_line+'\n')
                
                end_time_each = time.time()
                log_output.write(f'Processing time of category {category} is {end_time_each-start_time_each}.\n')


            end_time = time.time()
            log_output.write(f'Processing time of all categories is {end_time-start_time}.\n')

    
    def is_correct(self, response_txt, label_dir):
        exist_object = False
        if 'yes' in response_txt.lower():
            exist_object = True
        
        judgement = 'wrong'
        if ((not 'nonexist' in label_dir) and exist_object) or ('nonexist' in label_dir and not exist_object):
            judgement = 'correct'
        
        return judgement



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

parser = argparse.ArgumentParser(description='COCO-O robustness evaluation.')
parser.add_argument('--model_index', type=int, default=0, help='Model index to be evaluated.')
parser.add_argument('--device', type=str, default='cuda:0', help='Device.')
parser.add_argument('--root_dir', type=str, default='/data/zhangyichi/Trustworthy-MLLM/data/robustness/ood', help='Path to the input images.')
parser.add_argument('--output_dir', type=str, default='/data/zhangyichi/Trustworthy-MLLM/output/robustness/description_ood', help='Output dir.')
args = parser.parse_args()

model_index = args.model_index
model_name = model_list[model_index]
root_dir = args.root_dir
output_dir = args.output_dir

generation_kwargs = {
    'max_new_tokens': 200,
    'do_sample': False,
}

model = load_chatmodel(model_id=model_name, device=args.device)
test_engine=OODEval(args=args,
                    model=model,
                    model_name=model_name,
                    root_dir=root_dir,
                    out_dir=os.path.join(output_dir, model_name),
                    log_file=os.path.join(output_dir, model_name, 'alog.txt')
                    )
test_engine.eval(**generation_kwargs)
