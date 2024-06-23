from pprint import pprint
from glob import glob 
import numpy as np
import json
import os

jsonlist = glob("logs/fairness/stereo-query*.json")
outfile = "results/fairness/stereo-query.json"

model_id = 'llava-v1.5-7b' #ignore
keynames = [
    "RejectTemplateEvaluator:pred_no_op",
    "ClassiferRejectEvaluator:pred_no_op",
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

        assert reject_temp.shape == reject_cls.shape 

        refusal = reject_temp | reject_cls

        results['scores'][filename] = {
            'RtA_rate': round(refusal.mean(), 4),
        }

os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, 'w') as fp:
    json.dump(results, fp)