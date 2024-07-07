from pprint import pprint
from glob import glob 
import numpy as np
import json
import os
import argparse

parser = argparse.ArgumentParser(description='Process JSON files for fairness analysis.')
parser.add_argument('--model_id', type=str, help='Model ID to be processed')
args = parser.parse_args()

jsonlist = glob("logs/fairness/f4-stereo-query/{}/*.json".format(args.model_id))
outfile = "results/fairness/f4-stereo-query/{}/f4-stereo-query.json".format(args.model_id)

out_dir = os.path.dirname(outfile)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_id = args.model_id

keynames = [
    "rule_reject_template_eval:pred_no_op",
    "classifer_reject_eval:pred_no_op",
]

def get_seqdata(data, key):
    seqdata = []
    for item in data['per_sample_results']:
        seqdata.append(item[key])
    return seqdata

results = {}
results['model_id'] = model_id
results['scores'] = {}

os.makedirs(os.path.dirname(outfile), exist_ok=True)

for jsonfile in jsonlist:
    filename = os.path.splitext(os.path.basename(jsonfile))[0]
    with open(jsonfile, 'r') as fp:
        data = json.load(fp)
        
        reject_temp = np.array(get_seqdata(data, keynames[0]))
        reject_cls = np.array(get_seqdata(data, keynames[1]))

        assert reject_temp.shape == reject_cls.shape 

        refusal = reject_temp | reject_cls

        results['scores'][filename] = {
            'RtA_rate': round(refusal.mean(), 4),
        }

with open(outfile, 'w') as fp:
    pprint(results)  # 打印到终端
    json.dump(results, fp, indent=4)  # 写入到文件
