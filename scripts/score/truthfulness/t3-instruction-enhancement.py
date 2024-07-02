from pprint import pprint
from glob import glob 
import json
import os
import argparse

parser = argparse.ArgumentParser(description='Process JSON files for fairness analysis.')
parser.add_argument('--model_id', type=str, default='llava-v1.5-7b', help='Model ID to be processed')
args = parser.parse_args()


jsonlist = glob("logs/truthfulness/t3-instruction-enhancement/{}/*.json".format(args.model_id))
outfile = "results/truthfulness/t3-instruction-enhancement/{}/t3-instruction-enhancement.json".format(args.model_id)

out_dir = os.path.dirname(outfile)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_id = args.model_id
keynames = [
    "ChatModelEvaluator->YesOrNoEvaluator:pred_mean_PROMPT_1",
    "ChatModelEvaluator->YesOrNoEvaluator:pred_mean_PROMPT_2",
    "ChatModelEvaluator->YesOrNoEvaluator:pred_mean_PROMPT_3"
]


results = {}
results['model_id'] = model_id
results['scores'] = {}
for jsonfile in jsonlist:
    filename = os.path.splitext(os.path.basename(jsonfile))[0]
    with open(jsonfile, 'r') as fp:
        data = json.load(fp)

        results['scores'][filename] = {}
        varied_prompt_results = []
        for keyname in keynames:
            varied_prompt_results.append(format(data['total_results'][keyname]*100, '.2f'))
        results['scores'][filename]['Max_Accuracy'] = max(varied_prompt_results)
         
pprint(results)
os.makedirs(os.path.dirname(outfile), exist_ok=True)

with open(outfile, 'w') as fp:
    json.dump(results, fp, indent=4)