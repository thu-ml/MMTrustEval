from mmte.perspectives import get_task
from natsort import natsorted
from glob import glob
import json
import os

if __name__ == "__main__":
    lst = []
    lst += natsorted(glob('/data/zhangyichi/Trustworthy-MLLM/output/privacy/visual-leakage-vispr-debug/overlapped_bug/*.json'))
    lst += natsorted(glob('/data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-leakage-in-context-zeroshot-debug/*.json'))
    
    pre_task = ""
    for log_response in lst:
        task = os.path.basename(log_response).split('_on_')[0]
        print(f"Re-eval file: {log_response}")
        print(f"Task: {task}")
        
        if task != pre_task:
            task_handler = get_task(task)
        
        pre_task = task

        with open(log_response, "r") as f:
            log = json.load(f)

        responses = log['raw_log']
        result = task_handler.eval_response(responses)
        log['result'] = result

        with open(log_response, "w") as f:
            json.dump(log, f, indent=4)