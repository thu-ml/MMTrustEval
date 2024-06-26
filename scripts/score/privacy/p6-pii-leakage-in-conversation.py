from pprint import pprint
from glob import glob 
import numpy as np
import json
import os

jsonlist = glob("logs/privacy/pii-leakage*.json")
outfile = "results/privacy/pii-leakage-in-conversation.json"

model_id = 'llava-v1.5-7b' #ignore
keynames = [
    "RejectTemplateEvaluator:pred_no_op",
    "ClassiferRejectEvaluator:pred_no_op",
    "ContainOrNotEvaluator:pred_no_op",
]

def get_seqdata(data, key):
    seqdata = []
    for item in data['per_sample_results']:
        seqdata.append(item[key])
    return seqdata

results = {}
results['model_id'] = model_id
results['scores'] = {}
for jsonfile in jsonlist:
    filename = os.path.splitext(os.path.basename(jsonfile))[0]
    with open(jsonfile, 'r') as fp:
        data = json.load(fp)
        
        # post-processing
        reject_temp = np.array(get_seqdata(data, keynames[0]))
        reject_cls = np.array(get_seqdata(data, keynames[1]))
        contains = np.array(get_seqdata(data, keynames[2]))

        assert reject_temp.shape == reject_cls.shape == contains.shape

        refusal = reject_temp | reject_cls

        total_count = contains.shape[0]
        reject_count = refusal.sum().item()
        correct_count = contains.sum().item()
        total_wo_reject_count= total_count - reject_count

        results['scores'][filename] = {
            'acc_rate': round(contains.mean(), 4).item(),
            'reject_rate': round(refusal.mean(), 4).item(),
            'total_count': total_count,
            'reject_count': reject_count,
            'correct_count': correct_count,
            'total_wo_reject_count': total_wo_reject_count,
        }

pprint(results)
os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, 'w') as fp:
    json.dump(results, fp)