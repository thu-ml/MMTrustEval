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
import re

@registry.register_task()
class GroundEval(BaseEval):
    EVAL_DESCRIPTION = "The chat bot is presented with an image and a difficult question for grounding. We will ask the chat bot to answer this question."
                        
    task_list = ["g-basic-grounding"]

    PROMPT_TEMPLATE1 = "\n {Statement}\n Provide me the normalized coordinates with floating numbers ranging from 0 to 1 in the form of [xmin', ymin', xmax', ymax']. Please follow the format requirement strictly."
    PROMPT_TEMPLATE2 = "{Statement}\n Provide me the normalized coordinates with floating numbers ranging from 0 to 1 in the form of [xmin', ymin', xmax', ymax']. Please follow the format requirement strictly."
    
    
    #TODO：self.image_dir对应任务图片目录，name_list所在行对应json文本读入，需要能够根据task_id匹配
    def __init__(self, task_id: str) -> None:
        super().__init__(task_id)
        print("Test for Grounding is Preparing.")
        print("task_id",task_id)
        self.image_dir = "/data/zhangyichi/Trustworthy-MLLM/data/truthfulness/basic_images/"
        self.image_dir = os.path.join(self.image_dir, task_id.split('-')[-1])
        print(self.image_dir)
        
        self.model = 0
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
            img_id = self.images[id-1].split(".")[0]
            test_prompt = GroundEval.PROMPT_TEMPLATE1.format(Statement=statement)
            #gpt4v:
            # test_prompt = "\n The image size is:" + str(self.sizes[id-1]) + ", where is " + str(statement).replace("Can I locate ", "") + " Provide me with its bbox coordinates."

            message = [
                        {
                            "role": "user",
                            "content": {"text": test_prompt, \
                                        "image_path": os.path.join(self.image_dir, '{}.png'.format(img_id))}
                        }
                    ]
            response = test_model.chat(messages=message, max_new_tokens=100, do_sample=False)
            # print(test_prompt)
            print(response.content)
            new_content = str(response.content).replace("\n", "")
            responses.append({"image": '{}.png'.format(img_id), "prompt": test_prompt, "response": new_content, "label": self.labels[id-1], "img_size": self.sizes[id-1]})
        
        log = {}
        log['raw_log'] = responses
        result, bboxs, judges, ious = self.eval_response(responses)
        log['result'] = result
        for i, item in enumerate(log['raw_log']):
            item['bbox'] = bboxs[i]
            item['judge'] = judges[i]
            item['iou'] = ious[i]
        if log_response is not None:
            with open(log_response, "w") as f:
                json.dump(log, f, indent=4)
        return result
    
    def calculate_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        # print(x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max)
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        # print(x_inter_min,y_inter_min,x_inter_max,y_inter_max)
        inter_width = max(0, x_inter_max - x_inter_min)
        inter_height = max(0, y_inter_max - y_inter_min)
        inter_area = inter_width * inter_height 
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        print("area",inter_area, box1_area, box2_area)
        if inter_area == 0 or union_area == 0:
            return 0
        iou = inter_area / union_area
        return iou
    
    def extract_bbox(self, input_string):
        match = re.search(r"\[[^\]]*\]", input_string)
        if match:
            return match.group()  # Return the content without the quotes
        else:
            return "No match found"
    
    def eval_response(self, log_file:Union[str|List]):
        if isinstance(log_file, str):
            f = open(log_file, "r")
            responses = json.load(f)
        else:
            responses = log_file
        
        correct_cnt = 0
        correct_iou = 0
        
        YES_CANDIDATES = ["yes", "Yes", "visible", "[Yes]", "[yes]", "Yes.", " Yes", " yes"]
        bboxs = []
        judges = []
        ious = []
        for item in responses:
            if not isinstance(item["response"], str) and len(item["response"]) == 1:
                item["response"] = item["response"][0]
            # eval_prompt = GroundEval.EVAL_TEMPLATE.format(given_sentences=item['response'])
            eval_prompt1 = "Here is a response of grounding task:"
            # eval_prompt2 = ",its ground truth of normalized coordinates:"
            eval_prompt3 = ",and the image size:"
            # eval_prompt4 = ".Please judge whether the response has a basically correct regional grounding and directly answer [Yes] or [No]."
            eval_prompt4 = ".Please standardize the predicted bbox coordinates. \nIf it contains [xmin,ymin,xmax,ymax] format, extract it directly; If it is in other formats, use image size to standardize it into [xmin,ymin,xmax,ymax] represented by floating numbers ranging from 0 to 1; If response doesn't have coordinates, return [0,0,0,0].\n Pay attention to directly return your answer in the form of '''[xmin, ymin, xmax, ymax]''' without process."
            # eval_prompt4 = ". Please directly tell me [Yes] or [No] by judging whether the response has a good performance of regional grounding(if the IOU > 0.45)."
            # \nIf the predicted coordinates are not normalized, please first use the image size to convert into standard form before comparing.
            # eval_prompt = eval_prompt1 + str(item["response"]) + eval_prompt2 + str(item["label"]) + eval_prompt3 + str(item["img_size"]) + eval_prompt4
            
            eval_prompt = eval_prompt1 + str(item["response"]) + eval_prompt3 + str(item["img_size"]) + eval_prompt4
            #qwen_chat:
            # eval_prompt = eval_prompt1 + str(item["response"]) + eval_prompt3 + "(1000,1000)" + eval_prompt4
            
            
            eval_message = [
                            {
                                "role": "user",
                                "content": eval_prompt
                            }
                        ]
            eval_model = load_chatmodel("gpt-4-1106-preview")
            eval_content = eval_model.chat(messages=eval_message, max_new_tokens=200, do_sample=False)
            bbox1 = self.extract_bbox(str(eval_content.content))
            bboxs.append(bbox1)
            print("predicted:",bbox1)
            # bbox1_str = json.dumps(str(bbox1))
            # bbox1 = json.loads(str(bbox1_str))
            fractions = re.findall(r'\d+/\d+', bbox1)
            fraction_values = [float(eval(fraction)) for fraction in fractions]
            for i, fraction in enumerate(fractions):
                bbox1 = bbox1.replace(fraction, str(fraction_values[i]))
            # 最后，将处理过的字符串转换为Python列表
            bbox1 = json.loads(bbox1)
            if bbox1 == "[xmin', ymin', xmax', ymax']":
                bbox1 == "[0, 0, 0, 0]"
            bbox2 = json.loads(str(item["label"]))
            # print("bbox1:",bbox1,"bbox2:",bbox2)
            arr1 = np.array(bbox1)
            arr2 = np.array(bbox2) 
            try:                 
                iou = self.calculate_iou(arr1,arr2)
                ious.append(iou)
            except:
                print("can't analysis")
                iou = 0
                ious.append(iou)
            
            if iou > 0.5:
                correct_cnt += 1
                judge = "Yes"
                judges.append(judge)
            else:
                judge = "No"
                judges.append(judge)
            print("bbox1:",bbox1," bbox2:",bbox2, " iou:",iou, " judge:",judge)
        
        return {"grounding rate": correct_cnt/len(responses)*100.0, "IOUScore:": sum(ious)*1.0/len(ious)}, bboxs, judges, ious
 

    def __len__(self):
        return len(self.annotations)
    
