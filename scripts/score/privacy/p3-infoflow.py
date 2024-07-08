from pprint import pprint
from glob import glob 
import json
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Process JSON files for fairness analysis.')
parser.add_argument('--model_id', type=str, default='llava-v1.5-7b', help='Model ID to be processed')
args = parser.parse_args()


jsonlist = glob("logs/privacy/p3-infoflow/{}/*.json".format(args.model_id))
outfile = "results/privacy/p3-infoflow/{}/p3-infoflow.json".format(args.model_id)

out_dir = os.path.dirname(outfile)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_id = args.model_id

keyname_mapping = {
    "rule_match_and_score_eval:pearson_corr": "pearson_corr",
    "rule_match_and_score_eval:failure": "failure",
}

results = {}
results['model_id'] = model_id
results['scores'] = {}
for jsonfile in jsonlist:
    filename = os.path.splitext(os.path.basename(jsonfile))[0]
    with open(jsonfile, 'r') as fp:
        data = json.load(fp)

        results['scores'][filename] = {}
        for keyname in keyname_mapping.keys():
            newkeyname = keyname_mapping[keyname]
            results['scores'][filename][newkeyname] = round(data['total_results'][keyname], 4) if not np.isnan(data['total_results'][keyname]) else 0 

pprint(results)
os.makedirs(os.path.dirname(outfile), exist_ok=True)

with open(outfile, 'w') as fp:
    json.dump(results, fp, indent=4)