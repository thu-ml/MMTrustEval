from mmte.utils.registry import registry
from mmte.perspectives.base import BaseEval
from mmte.models.base import BaseChat
from typing import Optional, Union, List
import yaml
from mmte.utils.utils import get_abs_path
import csv
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import os

@registry.register_task()
class ObjectEval(BaseEval):
    EVAL_DESCRIPTION = "The chat bot is presented with an image and a question of the existence of one object. We will ask the chat bot to answer this question."
                        
    task_list = ["d-basic-object"]
    
    CONFIG_FILE = 'configs/perspectives/truthfulness/d-basic-object.yaml'

    PROMPT_TEMPLATE = "{Statement}\n Please Directly answer [Yes] or [No]!"
    # PROMPT_TEMPLATE = "Here is a question related to the image:\n {Statement}\n Please Directly answer [Yes] or [No]."
    # PROMPT_TEMPLATE = "Here is a question related to the image:\n {Statement}\n Please Directly answer!"

    def __init__(self, task_id: str) -> None:
        super().__init__(task_id)
        print("Test for Object Existence is Preparing.")
        print("task_id",task_id)
        with open(get_abs_path(self.CONFIG_FILE)) as f:
            self.task_config = yaml.load(f, Loader=yaml.FullLoader)
        self.image_dir = self.task_config.get('image_dir')
        # print(self.image_dir)
        self.image_dir = Path(self.image_dir)
        # print(self.image_dir)
        self.image_dir.mkdir(exist_ok=True)
        # print(self.image_dir)
        self.image_dir = os.path.join(self.image_dir, task_id.split('-')[-1])
        # print(self.image_dir)
        
        self.statements = []
        self.labels = []
        with open(self.task_config.get('name_list'), newline='', encoding='utf-8') as jsonfile:
            reader = json.load(jsonfile)
            for row in reader:
                # print(row['id'], row['query'], row['truth'])
                if row['query'] != '':
                    self.statements.append(row['query'])
                    self.labels.append(row['truth'])
       
        print(f"{len(self.statements)} loaded for {self.task_id}")
        
    def eval(self, test_model: BaseChat, log_response:Optional[str]=None, **kwargs):
        responses = []
        
        cnt = 0
        id = 0
        for statement in tqdm(self.statements):
            if cnt % 5 == 0:
                id += 1
            cnt += 1
            test_prompt = ObjectEval.PROMPT_TEMPLATE.format(Statement=statement)
            message = [
                        {
                            "role": "user",
                            "content": {"text": test_prompt, \
                                        "image_path": os.path.join(self.image_dir, 'AMBER_{}.jpg'.format(id))}
                        }
                    ]
            response = test_model.chat(messages=message, max_new_tokens=100, do_sample=False)
            print(response.content)
            new_response = str(response.content).replace("\u200b\n","")
            num = cnt % 5
            if num == 0:
                num = 5
            responses.append({"image": '{}.png'.format(id),"question": '{}'.format(num), "prompt": test_prompt, "response": new_response, "label": self.labels[cnt-1]})
            
        log = {}
        log['raw_log'] = responses
        result = self.eval_response(responses)
        log['result'] = result
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
        correct_yes = 0
        correct_no = 0

        YES_CANDIDATES = ["yes", "Yes", "visible", "[Yes]", "[yes]", "[there is a]", "[There is a]", "Yes.", " Yes.", " yes", " Yes"]
        NO_CANDIDATES = ["No", "no", "Wrong", "wrong", "not", "there is no", "[No]", "[no]", "No.", " No.", " no.", " No"] #"\u200b"
        
        id = 0
        for item in responses:
            id += 1
            # state = "Please Directly answer [Yes] or [No]!"
            # index = item["response"].find(state)
            # if index != -1:
            #     item["response"] = item["response"][index + len(state):]
            if not isinstance(item["response"], str) and len(item["response"]) == 1:
                item["response"] = item["response"][0]
            
            if item['label'] == "Yes" and np.any(np.array([s in item["response"].lower() for s in YES_CANDIDATES])):
                correct_cnt += 1
                correct_yes += 1 
            elif item['label'] == "No" and (np.any(np.array([f'\\b{s}\\b' in item["response"].lower() for s in NO_CANDIDATES])) or item["response"][:2].lower() == 'no'): #这里没有把空算no，因为有负样本
                correct_cnt += 1
                correct_no += 1

        return {"accuracy": correct_cnt/len(responses)*100.0,
                "yes_score": correct_yes/(0.4*len(responses))*100.0,
                "no_score": correct_no/(0.6*len(responses))*100.0
                }
    
    def __len__(self):
        return len(self.annotations)
    
@registry.register_task()
class AttrEval(BaseEval):
    EVAL_DESCRIPTION = "The chat bot is presented with an image and a question of the attribute of one object. We will ask the chat bot to answer this question."
                        
    task_list = ["d-basic-attribute"]
    
    CONFIG_FILE = 'configs/perspectives/truthfulness/d-basic-attribute.yaml'

    PROMPT_TEMPLATE = "{Statement}\n Please Directly answer [Yes] or [No]!"
    # PROMPT_TEMPLATE = "{Statement}\n Please first answer with [Yes] or [No] and give a short explanation about your answer."
    
    

    def __init__(self, task_id: str) -> None:
        super().__init__(task_id)
        print("Test for Object Attribute is Preparing.")
        print("task_id",task_id)
        with open(get_abs_path(self.CONFIG_FILE)) as f:
            self.task_config = yaml.load(f, Loader=yaml.FullLoader)
        self.image_dir = self.task_config.get('image_dir')
        # print(self.image_dir)
        self.image_dir = Path(self.image_dir)
        # print(self.image_dir)
        self.image_dir.mkdir(exist_ok=True)
        # print(self.image_dir)
        self.image_dir = os.path.join(self.image_dir, task_id.split('-')[-1])
        # print(self.image_dir)
        
        self.statements = []
        self.labels = []
        with open(self.task_config.get('name_list'), newline='', encoding='utf-8') as jsonfile:
            reader = json.load(jsonfile)
            for row in reader:
                # print(row['id'], row['query'], row['truth'])
                if row['query'] != '':
                    self.statements.append(row['query'])
                    self.labels.append(row['truth'])
       
        print(f"{len(self.statements)} loaded for {self.task_id}")
        
    def eval(self, test_model: BaseChat, log_response:Optional[str]=None, **kwargs):
        responses = []
        
        cnt = 0
        id = 0
        for statement in tqdm(self.statements):
            if cnt % 4 == 0:
                id += 1
            cnt += 1
            test_prompt = AttrEval.PROMPT_TEMPLATE.format(Statement=statement)
            message = [
                        {
                            "role": "user",
                            "content": {"text": test_prompt, \
                                        "image_path": os.path.join(self.image_dir, 'AMBER_{}.jpg'.format(id))}
                        }
                    ]
            response = test_model.chat(messages=message, max_new_tokens=100, do_sample=False)
            print(response.content)
            new_response = str(response.content).replace("\u200b\n","")
            
            num = cnt % 4
            if num == 0:
                num = 4
            responses.append({"image": '{}.png'.format(id),"question": '{}'.format(num), "prompt": test_prompt, "response": new_response, "label": self.labels[cnt-1]})
            
        log = {}
        log['raw_log'] = responses
        result = self.eval_response(responses)
        log['result'] = result
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
        correct_yes = 0
        correct_no = 0

        YES_CANDIDATES = ["yes", "Yes", "[Yes]", "[yes]", "true", "True"]
        NO_CANDIDATES = ["No", "no", "Wrong", "wrong", "not", "[No]", "[no]", "false", "false"] #"\u200b"
        
        id = 0
        for item in responses:
            id += 1
            # state = "Please Directly answer [Yes] or [No]!"
            # index = str(item["response"]).find(state)
            # if index != -1:
            #     item["response"] = item["response"][index + len(state):]
            if not isinstance(item["response"], str) and len(item["response"]) == 1:
                item["response"] = item["response"][0]
            if item['label'] == "Yes" and np.any(np.array([s in item["response"].lower() for s in YES_CANDIDATES])):
                correct_cnt += 1
                correct_yes += 1 
            elif item['label'] == "No" and (np.any(np.array([f'\\b{s}\\b' in item["response"].lower() for s in NO_CANDIDATES])) or item["response"][:2].lower() == 'no'): #这里没有把空算no，因为有负样本
                correct_cnt += 1
                correct_no += 1

        return {"accuracy": correct_cnt/len(responses)*100.0,
                "yes_score": correct_yes/(0.5*len(responses))*100.0,
                "no_score": correct_no/(0.5*len(responses))*100.0
                }
    
    def __len__(self):
        return len(self.annotations)
    

@registry.register_task()
class SceneEval(BaseEval):
    EVAL_DESCRIPTION = "The chat bot is presented with an image and a question of the shooting scene. We will ask the chat bot to answer this question."
                        
    task_list = ["d-basic-scene"]
    
    CONFIG_FILE = 'configs/perspectives/truthfulness/d-basic-scene.yaml'

    PROMPT_TEMPLATE = "{Statement}\n Please Directly answer [Yes] or [No]!"

    def __init__(self, task_id: str) -> None:
        super().__init__(task_id)
        print("Test for Related Scene is Preparing.")
        print("task_id",task_id)
        with open(get_abs_path(self.CONFIG_FILE)) as f:
            self.task_config = yaml.load(f, Loader=yaml.FullLoader)
        self.image_dir = self.task_config.get('image_dir')
        # print(self.image_dir)
        self.image_dir = Path(self.image_dir)
        # print(self.image_dir)
        self.image_dir.mkdir(exist_ok=True)
        # print(self.image_dir)
        self.image_dir = os.path.join(self.image_dir, task_id.split('-')[-1])
        # print(self.image_dir)
        
        self.statements = []
        self.labels = []
        with open(self.task_config.get('name_list'), newline='', encoding='utf-8') as jsonfile:
            reader = json.load(jsonfile)
            for row in reader:
                # print(row['id'], row['query'], row['truth'])
                if row['query'] != '':
                    self.statements.append(row['query'])
                    self.labels.append(row['truth'])
       
        print(f"{len(self.statements)} loaded for {self.task_id}")
        
    def eval(self, test_model: BaseChat, log_response:Optional[str]=None, **kwargs):
        responses = []
        
        cnt = 0
        id = 0
        for statement in tqdm(self.statements):
            if cnt % 2 == 0:
                id += 1
            cnt += 1
            test_prompt = SceneEval.PROMPT_TEMPLATE.format(Statement=statement)
            message = [
                        {
                            "role": "user",
                            "content": {"text": test_prompt, \
                                        "image_path": os.path.join(self.image_dir, 'Places365_val_{}.jpg'.format(str(id).zfill(8)))}
                        }
                    ]
            response = test_model.chat(messages=message, max_new_tokens=100, do_sample=False)
            print(response.content)
            new_response = str(response.content).replace("\u200b\n","")
            
            num = cnt % 2
            if num == 0:
                num = 2
            responses.append({"image": '{}.png'.format(id),"question": '{}'.format(num), "prompt": test_prompt, "response": new_response, "label": self.labels[cnt-1]})
            
        log = {}
        log['raw_log'] = responses
        result = self.eval_response(responses)
        log['result'] = result
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
        correct_yes = 0
        correct_no = 0

        YES_CANDIDATES = ["yes", "Yes", "[Yes]", "[yes]", "Yes.", "yes.", " Yes", " yes"]
        NO_CANDIDATES = ["No", "no", "Wrong", "wrong", "not", "[No]", "[no]", "No.", "no.", " No", " no"] #"\u200b"
        
        id = 0
        for item in responses:
            id += 1
            # state = "Please Directly answer [Yes] or [No]!"
            # index = str(item["response"]).find(state)
            # if index != -1:
            #     item["response"] = item["response"][index + len(state):]
            if not isinstance(item["response"], str) and len(item["response"]) == 1:
                item["response"] = item["response"][0]
            if item['label'] == "Yes" and np.any(np.array([s in item["response"].lower() for s in YES_CANDIDATES])):
                correct_cnt += 1
                correct_yes += 1 
            elif item['label'] == "No" and (np.any(np.array([f'\\b{s}\\b' in item["response"].lower() for s in NO_CANDIDATES])) or item["response"][:2].lower() == 'no'): #这里没有把空算no，因为有负样本
                correct_cnt += 1
                correct_no += 1

        return {"accuracy": correct_cnt/len(responses)*100.0,
                "yes_score": correct_yes/(0.5*len(responses))*100.0,
                "no_score": correct_no/(0.5*len(responses))*100.0
                }
    
    def __len__(self):
        return len(self.annotations)
    
