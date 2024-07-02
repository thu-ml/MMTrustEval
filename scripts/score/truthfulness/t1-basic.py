from pprint import pprint
from glob import glob 
import json
import os
import argparse

parser = argparse.ArgumentParser(description='Process JSON files for fairness analysis.')
parser.add_argument('--model_id', type=str, default='llava-v1.5-7b', help='Model ID to be processed')
args = parser.parse_args()


jsonlist = glob("logs/truthfulness/t1-basic/{}/*.json".format(args.model_id))
outfile = "results/truthfulness/t1-basic/{}/t1-basic-world-understanding.json".format(args.model_id)

out_dir = os.path.dirname(outfile)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_id = args.model_id
keynames = [
    "YesOrNoEvaluator:accuracy_score",
    "ChatModelEvaluator:iou_judge",
    "ChatModelEvaluator->YesOrNoEvaluator:pred_mean"
]

results = {}
results['model_id'] = model_id
results['scores'] = {}
for jsonfile in jsonlist:
    filename = os.path.splitext(os.path.basename(jsonfile))[0]
    with open(jsonfile, 'r') as fp:
        data = json.load(fp)

        results['scores'][filename] = {}
        if filename == 'd-basic-object':
            newkeyname = 'Accuracy_Object'
            results['scores'][filename][newkeyname] = format(data['total_results'][keynames[0]]*100, '.2f')
        elif filename == 'd-basic-attribute':
            newkeyname = 'Accuracy_Attribute'
            results['scores'][filename][newkeyname] = format(data['total_results'][keynames[0]]*100, '.2f')     
        elif filename == 'd-basic-scene':
            newkeyname = 'Accuracy_Scene'
            results['scores'][filename][newkeyname] = format(data['total_results'][keynames[0]]*100, '.2f')  
        elif filename == 'g-basic-grounding':
            newkeyname = 'Accuracy_Grounding'
            results['scores'][filename][newkeyname] = format(data['total_results'][keynames[1]], '.2f')  
        elif filename == 'g-basic-ocr':
            newkeyname = 'Accuracy_OCR'
            results['scores'][filename][newkeyname] = format(data['total_results'][keynames[2]]*100, '.2f')        

pprint(results)
os.makedirs(os.path.dirname(outfile), exist_ok=True)

with open(outfile, 'w') as fp:
    json.dump(results, fp, indent=4)