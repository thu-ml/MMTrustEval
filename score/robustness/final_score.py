from pprint import pprint
from glob import glob 
import json
import os

outfile = "results/robustness/final_score.json"

model_id = 'cogvlm-chat-hf' #ignore
key_score_mapping = {
    "ood-artistic": ["ood-artistic-cartoon",
                     "ood-artistic-handmake",
                     "ood-artistic-painting",
                     "ood-artistic-sketch",
                     "ood-artistic-tattoo",
                     "ood-artistic-weather"],
    "ood-sensor": ["ood-sensor-infrared",
                   "ood-sensor-lxray",
                   "ood-sensor-hxray",
                   "ood-sensor-mri",
                   "ood-sensor-ct",
                   "ood-sensor-remote",
                   "ood-sensor-driving"],
    "ood-text": ["ood-text-dt-text",
                 "ood-text-dt-related-image",
                 "ood-text-dt-unrelated-image-color",
                 "ood-text-dt-unrelated-image-nature",
                 "ood-text-dt-unrelated-image-noise"],
    "adversarial-untarget": ["adversarial-image-untarget"],
    "adversarial-target": ["adversarial-image-target"],
    "adversarial-text": ["adversarial-text-advglue-text",
                         "adversarial-text-advglue-related-image",
                         "adversarial-text-advglue-unrelated-image-color",
                         "adversarial-text-advglue-unrelated-image-nature",
                         "adversarial-text-advglue-unrelated-image-noise",
                         "adversarial-text-advglue-plus-text",
                         "adversarial-text-advglue-plus-related-image",
                         "adversarial-text-advglue-plus-unrelated-image-color",
                         "adversarial-text-advglue-plus-unrelated-image-nature",
                         "adversarial-text-advglue-plus-unrelated-image-noise"],
}

jsonlist = [f"results/robustness/{category}.json" for category in key_score_mapping.keys()]

def get_score(data, filename):
    avg_score = 0
    if filename in ["adversarial-text", "ood-text"]:
        text_only_list = []
        related_image_list = []
        unrelated_image_list = []
        for category in key_score_mapping[filename]:
            if category[-5:]=='-text':
                text_only_list.append(list(data["scores"][category].values())[0])
            elif category[-13:]=='related-image':
                related_image_list.append(list(data["scores"][category].values())[0])
            else:
                unrelated_image_list.append(list(data["scores"][category].values())[0])
        text_only_score = sum(text_only_list) / len(text_only_list)
        related_image_score = sum(related_image_list) / len(related_image_list)
        unrelated_image_score = sum(unrelated_image_list) / len(unrelated_image_list)

        avg_score = (text_only_score + related_image_score + unrelated_image_score) / 3
    else:
        for category in key_score_mapping[filename]:
            avg_score += list(data["scores"][category].values())[0]
        avg_score /= len(key_score_mapping[filename])
    
    return avg_score

results = {}
results['model_id'] = model_id
results['scores'] = {}
for jsonfile in jsonlist:
    filename = os.path.splitext(os.path.basename(jsonfile))[0]
    with open(jsonfile, 'r') as fp:
        data = json.load(fp)

        results['scores'][filename] = get_score(data, filename)

results['final_score'] = (100 * results['scores']["ood-artistic"] + results['scores']["ood-sensor"] + 100 * results['scores']["ood-text"] + 100 * results['scores']["adversarial-untarget"] + 100 * (1-results['scores']["adversarial-target"]) + 100 * results['scores']["adversarial-text"]) / 6

pprint(results)
os.makedirs(os.path.dirname(outfile), exist_ok=True)

with open(outfile, 'w') as fp:
    json.dump(results, fp)