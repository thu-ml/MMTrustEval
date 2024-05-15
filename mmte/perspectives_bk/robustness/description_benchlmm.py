import sys
sys.path.append('/data/zhangyichi/Trustworthy-MLLM/MMTrustEval')

import os
import json
import argparse
import mmte
from mmte.models import load_chatmodel, model_zoo
import time
from PIL import Image

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
            # 'infrard': {'json_file': 'json/Benchmark_infrard.jsonl', 'image_dir': 'source/infrard'},
            # 'lxray': {'json_file': 'json/Benchmark_xray.jsonl', 'image_dir': 'source/Xray'},
            # 'hxray': {'json_file': 'json/Benchmark_Med-X-RAY.jsonl', 'image_dir': 'source/Med'},
            # 'mri': {'json_file': 'json/Benchmark_MRI.jsonl', 'image_dir': 'source/Med'},
            # 'ct': {'json_file': 'json/Benchmark_CT.jsonl', 'image_dir': 'source/Med'},
            'remote': {'json_file': 'json/Benchmark_RS.jsonl', 'image_dir': 'source/RS'},
            'driving': {'json_file': 'json/Benchmark_AD.jsonl', 'image_dir': 'source/AD'},
            'robot': {'json_file': 'json/Benchmark_Robots.jsonl', 'image_dir': 'source/Robots'},
            'game': {'json_file': 'json/Benchmark_game.jsonl', 'image_dir': 'source/open_game'},
            'defect': {'json_file': 'json/Benchmark_defect_detection.jsonl', 'image_dir': 'source/defect_detection'},
            'style_cartoon': {'json_file': 'json/Benchmark_style_cartoon.jsonl', 'image_dir': 'source/style'},
            # 'style_handmake': {'json_file': 'json/Benchmark_style_handmake.jsonl', 'image_dir': 'source/style'},
            # 'style_painting': {'json_file': 'json/Benchmark_style_painting.jsonl', 'image_dir': 'source/style'},
            # 'style_sketch': {'json_file': 'json/Benchmark_style_sketch.jsonl', 'image_dir': 'source/style'},
            # 'style_tattoo': {'json_file': 'json/Benchmark_style_tattoo.jsonl', 'image_dir': 'source/style'},
        }
        self.tmp_dir='/data/zhangyichi/Trustworthy-MLLM/data/robustness/temp'
    
    def eval(self, **generation_kwargs):
        print(f'Processing model {self.model_name}.')
        start_time = time.time()
        with open(self.log_file, 'w') as log_output:
            for category, category_info in self.categories.items():
                start_time_each = time.time()

                # input data
                test_data_dict={}
                with open(os.path.join(self.root_dir, category_info['json_file']), 'r') as f:
                    for line in f.readlines():
                        json_line=json.loads(line.strip())
                        if json_line['answer']=='None' or json_line['answer']=='A: None':
                            continue
                        test_data_dict[json_line['question_id']]=json_line
                
                # evaluation
                response_list=[]
                for image_id, image_info in test_data_dict.items():
                    try:
                        image_name=image_info['image']
                        text=image_info['text']
                        answer=image_info['answer']

                        image_path=os.path.join(self.root_dir, category_info['image_dir'], image_name)

                        if self.tmp_dir is not None:
                            input_image=Image.open(image_path)
                            (width, height) = input_image.size
                            if width>500 or height>500:
                                resize_ratio=500/max(width, height)
                            input_image = input_image.resize((int(width*resize_ratio), int(height*resize_ratio)), Image.Resampling.LANCZOS)

                            image_path=os.path.join(self.tmp_dir, category_info['image_dir'], image_name)
                            dir_name,_=os.path.split(image_path)
                            os.makedirs(dir_name, exist_ok=True)
                            input_image.save(image_path)
                        
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
                        break

                    response_list.append({'image_id': image_id, 'image_name': image_name, 'question': text, 'gt_caption': answer, 'gt_label': None, 'response': response_txt})
                    
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

parser = argparse.ArgumentParser(description='BenchLMM robustness evaluation.')
parser.add_argument('--model_index', type=int, default=0, help='Model index to be evaluated.')
parser.add_argument('--device', type=str, default='cuda:0', help='Device.')
parser.add_argument('--root_dir', type=str, default='/data/zhangyichi/Trustworthy-MLLM/data/robustness/benchlmm', help='Path to the input images.')
parser.add_argument('--output_dir', type=str, default='/data/zhangyichi/Trustworthy-MLLM/output/robustness/description_benchlmm', help='Output dir.')
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
