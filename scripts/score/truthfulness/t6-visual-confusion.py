import argparse
import json
import os
from glob import glob
from pprint import pprint

parser = argparse.ArgumentParser(description='Process JSON files for fairness analysis.')
parser.add_argument('--model_id', type=str, default='llava-v1.5-7b', help='Model ID to be processed')
args = parser.parse_args()


jsonlist = glob("logs/truthfulness/t6-visual-confusion/{}/*.json".format(args.model_id))
outfile = "results/truthfulness/t6-visual-confusion/{}/t6-visual-confusion.json".format(args.model_id)

out_dir = os.path.dirname(outfile)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_id = args.model_id
keyname_mapping = {
    "rule_yes_or_no_eval:accuracy_score_semantic": 'Accuracy_Semantic',
    "rule_yes_or_no_eval:accuracy_score_contrast": 'Accuracy_Contrast',
    "rule_yes_or_no_eval:accuracy_score_painting": 'Accuracy_Painting',
    "rule_yes_or_no_eval:accuracy_score_mirror": 'Accuracy_Mirror',
    "rule_yes_or_no_eval:accuracy_score_dislocation": 'Accuracy_Dislocation',
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
            results['scores'][filename][newkeyname] = format(data['total_results'][keyname]*100, '.2f')
         
pprint(results)
os.makedirs(os.path.dirname(outfile), exist_ok=True)

with open(outfile, 'w') as fp:
    json.dump(results, fp, indent=4)