import argparse
from glob import glob
import json
import os
from pprint import pprint

parser = argparse.ArgumentParser(description='Process JSON files for fairness analysis.')
parser.add_argument('--model_id', type=str, help='Model ID to be processed')
args = parser.parse_args()

jsonlist = glob("logs/fairness/f3-stereo-classification/{}/stereo-classification*.json".format(args.model_id))
outfile = "results/fairness/f3-stereo-classification/{}/f3-stereo-classification.json".format(args.model_id)

out_dir = os.path.dirname(outfile)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_id = args.model_id  

keyname_mapping = {
    "rule_contain_eval:pred_mean": "Accuracy_Average",
    "rule_contain_eval:pred_mean_stereotype": "Accuracy_Stereotype",
    "rule_contain_eval:pred_mean_anti-stereotype": "Accuracy_Anti_Stereotype",
    "rule_contain_eval:pred_mean_unrelated": "Accuracy_Unrelated",
}

results = {}
results['model_id'] = model_id
results['scores'] = {}

os.makedirs(os.path.dirname(outfile), exist_ok=True)

with open(outfile, 'w') as fp:
    for jsonfile in jsonlist:
        filename = os.path.splitext(os.path.basename(jsonfile))[0]
        with open(jsonfile, 'r') as data_file:
            data = json.load(data_file)

            file_scores = {}
            for keyname, newkeyname in keyname_mapping.items():
                file_scores[newkeyname] = round(data['total_results'][keyname], 4)
            
            results['scores'][filename] = file_scores

    pprint(results)
    json.dump(results, fp, indent=4)
    fp.write("\n")
