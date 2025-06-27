import os
import subprocess


def find_files_by_name(directory, name):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filename = os.path.join(directory, file)
            if name in filename:
                matched_files.append(filename)

        for dir in dirs:
            matched_files.extend(find_files_by_name(os.path.join(directory, dir), name))
    return matched_files


if __name__ == "__main__":
    # directory = "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs_bk"
    # name = "cambrian-8b"
    # directory = '/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs'
    # name = "glm-4v-chat"
    name = "deepseek-vl2"
    # name = "deepseek-janus-7b"

    # matched_files = find_files_by_name(directory, name)
    # for file in matched_files:
    #     print(file)
    # print(len(matched_files))

    path_templates = [
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t5-text-misleading/{}/g-text-misleading.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t6-visual-confusion/{}/d-mis-visual-confusion.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t4-visual-assistance/{}/visual-assistance-image.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t4-visual-assistance/{}/visual-assistance-unrelated-image-nature.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t4-visual-assistance/{}/visual-assistance-text.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t4-visual-assistance/{}/visual-assistance-unrelated-image-color.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t4-visual-assistance/{}/visual-assistance-unrelated-image-noise.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t2-advanced/{}/advanced-translate.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t2-advanced/{}/advanced-math.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t2-advanced/{}/advanced-traffic.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t2-advanced/{}/advanced-spatial.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t2-advanced/{}/advanced-compare.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t2-advanced/{}/advanced-daily.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t2-advanced/{}/advanced-temporal.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t2-advanced/{}/advanced-code.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t2-advanced/{}/advanced-causality.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t7-visual-misleading/{}/g-text-unrelated-image-nature.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t7-visual-misleading/{}/g-text-unrelated-image-color.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t7-visual-misleading/{}/g-text-assistance.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t7-visual-misleading/{}/g-text-misvisual.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t7-visual-misleading/{}/g-text-none.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t7-visual-misleading/{}/g-text-unrelated-image-noise.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t1-basic/{}/d-basic-object.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t1-basic/{}/d-basic-attribute.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t1-basic/{}/g-basic-grounding.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t1-basic/{}/d-basic-scene.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t1-basic/{}/g-basic-ocr.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t3-instruction-enhancement/{}/instruction-enhancement-factual.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/truthfulness/t3-instruction-enhancement/{}/instruction-enhancement-logic.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f2-stereo-agreement/{}/stereo-agreement-unrelated-image-color.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f2-stereo-agreement/{}/stereo-agreement-unrelated-image-nature.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f2-stereo-agreement/{}/stereo-agreement-image.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f2-stereo-agreement/{}/stereo-agreement-unrelated-image-noise.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f2-stereo-agreement/{}/stereo-agreement-text.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f6-profession-pred/{}/profession-pred-with-description.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f6-profession-pred/{}/profession-pred.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f4-stereo-query/{}/stereo-query-unrelated-image-color.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f4-stereo-query/{}/stereo-query-image.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f4-stereo-query/{}/stereo-query-unrelated-image-nature.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f4-stereo-query/{}/stereo-query-text.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f4-stereo-query/{}/stereo-query-unrelated-image-noise.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f3-stereo-classification/{}/stereo-classification-image.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f3-stereo-classification/{}/stereo-classification-unrelated-image-nature.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f3-stereo-classification/{}/stereo-classification-text.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f3-stereo-classification/{}/stereo-classification-unrelated-image-color.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f3-stereo-classification/{}/stereo-classification-unrelated-image-noise.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f5-vision-preference/{}/vision-preference.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f7-subjective-preference/{}/subjective-preference-force-unrelated-image-noise.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f7-subjective-preference/{}/subjective-preference-force-unrelated-image-nature.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f7-subjective-preference/{}/subjective-preference-plain-image.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f7-subjective-preference/{}/subjective-preference-plain-text.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f7-subjective-preference/{}/subjective-preference-force-unrelated-image-color.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f7-subjective-preference/{}/subjective-preference-force-image.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f7-subjective-preference/{}/subjective-preference-plain-unrelated-image-nature.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f7-subjective-preference/{}/subjective-preference-plain-unrelated-image-noise.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f7-subjective-preference/{}/subjective-preference-plain-unrelated-image-color.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f7-subjective-preference/{}/subjective-preference-force-text.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f3-stereo-topic-classification/{}/stereo-topic-classification-unrelated-image-noise.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f3-stereo-topic-classification/{}/stereo-topic-classification-unrelated-image-nature.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f3-stereo-topic-classification/{}/stereo-topic-classification-unrelated-image-color.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f3-stereo-topic-classification/{}/stereo-topic-classification-text.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f3-stereo-topic-classification/{}/stereo-topic-classification-image.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/fairness/f1-stereo-generation/{}/stereo-generation.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r6-adversarial-text/{}/advglue-unrelated-image-color.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r6-adversarial-text/{}/advglue-plus-unrelated-image-nature.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r6-adversarial-text/{}/advglue-unrelated-image-nature.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r6-adversarial-text/{}/advglue-plus-unrelated-image-color.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r6-adversarial-text/{}/advglue-related-image.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r6-adversarial-text/{}/advglue-unrelated-image-noise.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r6-adversarial-text/{}/advglue-plus-unrelated-image-noise.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r6-adversarial-text/{}/advglue-text.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r6-adversarial-text/{}/advglue-plus-text.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r6-adversarial-text/{}/advglue-plus-related-image.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r5-adversarial-target/{}/adv-target.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r2-ood-sensor/{}/benchlmm-driving.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r2-ood-sensor/{}/benchlmm-hxray.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r2-ood-sensor/{}/benchlmm-lxray.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r2-ood-sensor/{}/benchlmm-mri.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r2-ood-sensor/{}/benchlmm-infrared.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r2-ood-sensor/{}/benchlmm-ct.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r2-ood-sensor/{}/benchlmm-remote.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r1-ood-artistic/{}/coco-o-weather.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r1-ood-artistic/{}/coco-o-tattoo.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r1-ood-artistic/{}/coco-o-painting.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r1-ood-artistic/{}/coco-o-sketch.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r1-ood-artistic/{}/coco-o-handmake.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r1-ood-artistic/{}/coco-o-cartoon.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r4-adversarial-untarget/{}/adv-clean.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r4-adversarial-untarget/{}/adv-untarget.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r3-ood-text/{}/dt-related-image.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r3-ood-text/{}/dt-text.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r3-ood-text/{}/dt-unrelated-image-nature.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r3-ood-text/{}/dt-unrelated-image-color.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/robustness/r3-ood-text/{}/dt-unrelated-image-noise.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s5-multimodal-jailbreaking/{}/safebench.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s5-multimodal-jailbreaking/{}/opimized-jailbreak-graphic.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s5-multimodal-jailbreaking/{}/mm-safety-bench.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s6-crossmodal-jailbreaking/{}/crossmodal-jailbreak-text.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s6-crossmodal-jailbreaking/{}/crossmodal-jailbreak-neg.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s6-crossmodal-jailbreaking/{}/crossmodal-jailbreak-unrelated.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s6-crossmodal-jailbreaking/{}/crossmodal-jailbreak-pos.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s4-typographic-jailbreaking/{}/typographic-prompt.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s4-typographic-jailbreaking/{}/typographic-behavior.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s4-typographic-jailbreaking/{}/typographic-prompt-and-behavior.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s1-nsfw-image-description/{}/nsfw-image-description.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s2-risk-identification/{}/risk-analysis.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s2-risk-identification/{}/object-detection.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s3-toxic-content-generation/{}/toxicity-prompt-text.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s3-toxic-content-generation/{}/toxicity-prompt-unrelated.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/safety/s3-toxic-content-generation/{}/toxicity-prompt-image.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p5-visual-leakage/{}/vispr-leakage-protected.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p5-visual-leakage/{}/vispr-leakage.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p4-pii-query/{}/celebrities-personal-phone-name-occupation.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p4-pii-query/{}/celebrities-personal-email-wo-name-wo-occupation.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p4-pii-query/{}/celebrities-personal-address-name-occupation.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p4-pii-query/{}/celebrities-personal-email-name-wo-occupation.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p4-pii-query/{}/celebrities-personal-address-wo-name-occupation.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p4-pii-query/{}/celebrities-personal-address-name-wo-occupation.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p4-pii-query/{}/celebrities-personal-address-wo-name-wo-occupation.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p4-pii-query/{}/celebrities-personal-phone-name-wo-occupation.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p4-pii-query/{}/celebrities-personal-email-name-occupation.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p4-pii-query/{}/celebrities-personal-phone-wo-name-wo-occupation.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p4-pii-query/{}/celebrities-personal-phone-wo-name-occupation.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p4-pii-query/{}/celebrities-personal-email-wo-name-occupation.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p6-pii-leakage-in-conversation/{}/enron-email-unrelated-image-color.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p6-pii-leakage-in-conversation/{}/enron-email-image-typeinfo.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p6-pii-leakage-in-conversation/{}/enron-email-image-allinfo.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p6-pii-leakage-in-conversation/{}/enron-email-unrelated-image-noise.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p6-pii-leakage-in-conversation/{}/enron-email-text.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p6-pii-leakage-in-conversation/{}/enron-email-unrelated-image-nature.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p6-pii-leakage-in-conversation/{}/enron-email-image-oneinfo.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p3-infoflow/{}/confaide-unrelated-image-color.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p3-infoflow/{}/confaide-unrelated-image-noise.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p3-infoflow/{}/confaide-unrelated-image-nature.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p3-infoflow/{}/confaide-text.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p3-infoflow/{}/confaide-image.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p2-vqa-recognition-vispr/{}/vispr-recognition-pri-query.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p1-vispriv-recognition/{}/vizwiz-recognition.json",
        "/data/zhangyichi/MMTrustEval-dev/dev-new-models/logs/privacy/p1-vispriv-recognition/{}/vispr-recognition.json",
    ]

    no_exists = []
    for template in path_templates:
        path = template.format(name)
        if os.path.exists(path):
            continue

        no_exists.append(path)

    print(no_exists)

    for no_exist in no_exists:
        filename = os.path.basename(no_exist).split(".")[0]
        result = subprocess.run(
            ["find", "run_logs", "-name", f"*{name}*{filename}*"],
            capture_output=True,
            text=True,
        )

        print(result.stdout.strip())
