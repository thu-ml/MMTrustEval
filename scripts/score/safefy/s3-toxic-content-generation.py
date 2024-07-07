from pprint import pprint
from glob import glob 
import json
import os
import argparse

parser = argparse.ArgumentParser(description='Process JSON files for fairness analysis.')
parser.add_argument('--model_id', type=str, default='llava-v1.5-7b', help='Model ID to be processed')
args = parser.parse_args()


jsonlist = glob("logs/safety/s3-toxic-content-generation/{}/*.json".format(args.model_id))
outfile = "results/safety/s3-toxic-content-generation/{}/s3-toxic-content-generation.json".format(args.model_id)

out_dir = os.path.dirname(outfile)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_id = args.model_id

keyname_mapping = {
    "google_perspective_api_eval:pred_mean": "PersepectiveScore",
    "openai_moderation_api_eval:pred_mean": "ModerationScore",
    "rule_reject_template_eval:pred_mean": "RtA",
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
            results['scores'][filename][newkeyname] = round(data['total_results'][keyname], 4)

pprint(results)
os.makedirs(os.path.dirname(outfile), exist_ok=True)

with open(outfile, 'w') as fp:
    json.dump(results, fp, indent=4)