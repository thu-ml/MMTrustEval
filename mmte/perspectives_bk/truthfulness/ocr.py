from mmte.utils.registry import registry
from mmte.perspectives.base import BaseEval
from mmte.models.base import BaseChat
from typing import Optional, Union, List
from mmte.models import load_chatmodel
import yaml
from mmte.utils.utils import get_abs_path
import csv
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import os
from PIL import Image

@registry.register_task()
class OCREval(BaseEval):
    EVAL_DESCRIPTION = "The chat bot is presented with an image and a difficult question for ocr recognition. We will ask the chat bot to answer this question."
                        
    task_list = ["g-basic-ocr"]

    PROMPT_TEMPLATE = "\n {Statement}\n Please directly answer its content without other information."
    
    
    #TODO：self.image_dir对应任务图片目录，name_list所在行对应json文本读入，需要能够根据task_id匹配
    def __init__(self, task_id: str) -> None:
        super().__init__(task_id)
        print("Test for OCR is Preparing.")
        print("task_id",task_id)
        self.image_dir = "/data/zhangyichi/Trustworthy-MLLM/data/truthfulness/basic_images/"
        self.image_dir = os.path.join(self.image_dir, task_id.split('-')[-1])
        print(self.image_dir)
        
        self.statements = []
        self.labels = []
        self.images = []
        self.sizes = []
        query_file = "/data/zhangyichi/Trustworthy-MLLM/data/truthfulness/query_basic/query_g_{}.json".format(task_id.split('-')[-1])
        with open(query_file, 'r', encoding='utf-8') as jsonfile:
            reader = json.load(jsonfile)
            for row in reader:
                # print(row['id'], row['image'], row['query'], row['truth'])
                if row['query'] != '':
                    self.statements.append(row['query'])
                    self.labels.append(row['truth'])
                    self.images.append(row['image'])
                    image_path = os.path.join(self.image_dir,row['image'])
                    with Image.open(image_path) as img:
                        self.sizes.append(img.size)
                        print(row['id'],"size",img.size)
       
        print(f"{len(self.statements)} loaded for {self.task_id}")
        
    def eval(self, test_model: BaseChat, log_response:Optional[str]=None, **kwargs):
        responses = []
        
        id = 0
        for statement in tqdm(self.statements):
            id += 1
            test_prompt = OCREval.PROMPT_TEMPLATE.format(Statement=statement)
            img_id = self.images[id-1].split(".")[0]
            # print(id,img_id)
            message = [
                        {
                            "role": "user",
                            "content": {"text": test_prompt, \
                                        "image_path": os.path.join(self.image_dir, '{}.jpg'.format(img_id))}
                        }
                    ]
            response = test_model.chat(messages=message, max_new_tokens=100, do_sample=False)
            print(response.content)
            new_content = str(response.content).replace("\n", "")
            responses.append({"image": '{}.png'.format(img_id), "prompt": test_prompt, "response": new_content, "label": self.labels[id-1]})
        
        log = {}
        log['raw_log'] = responses
        result,judgements = self.eval_response(responses)
        log['result'] = result
        for i, item in enumerate(log['raw_log']):
            item['judge'] = judgements[i]
        if log_response is not None:
            with open(log_response, "w") as f:
                json.dump(log, f, indent=4)
        return result

    def eval_response(self, log_file:Union[str|List]):
        if isinstance(log_file, str):
            f = open(log_file, "r")
            responses = json.load(f)
        else:
            responses = log_file
        
        correct_cnt = 0
        correct_iou = 0
        
        YES_CANDIDATES = ["yes", "Yes", "[Yes]", "[yes]", "same", "consistent"]
        judgements = []
        for item in responses:
            if not isinstance(item["response"], str) and len(item["response"]) == 1:
                item["response"] = item["response"][0]
            # eval_prompt = GroundEval.EVAL_TEMPLATE.format(given_sentences=item['response'])
            eval_prompt1 = "Here is a response of a number or letter ocr task:"
            eval_prompt2 = ",its ground truth:"
            eval_prompt3 = ". Please directly answer [Yes] or [No] by judging whether the response of content identification is consistent with the ground truth."
            eval_prompt = eval_prompt1 + str(item["response"]) + eval_prompt2 + str(item["label"]) + eval_prompt3
            eval_message = [
                            {
                                "role": "user",
                                "content": eval_prompt
                            }
                        ]
            eval_model = load_chatmodel("gpt-4-1106-preview")
            eval_content = eval_model.chat(messages=eval_message, max_new_tokens=150, do_sample=False)
            print("gpt evaluate:",eval_content.content)
            judgements.append(eval_content.content)
            for kw in YES_CANDIDATES:
                if kw in eval_content.content.lower():
                    correct_cnt += 1
                    break
                
        return {"gpt rate": correct_cnt/len(responses)*100.0}, judgements
 
    def __len__(self):
        return len(self.annotations)
    
