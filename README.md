
<h2 align="center">Benchmarking Trustworthiness of Multimodal Large Language Models:<br>
A Comprehensive Study
</h2>

<font size=3>
    <p align="center"> This is the official repository of <b>MMTrustEval</b>, the toolbox for conducting benchmarks on trustworthiness of MLLMs (<b>MultiTrust</b>) </p>
</font>

<div align="center" style="font-size: 16px;">
    🍎 <a href="https://multi-trust.github.io/">Project Page</a> &nbsp&nbsp
    📖 <a href="https://arxiv.org/abs/2406.07057">arXiv Paper</a> &nbsp&nbsp
    📊 <a href="https://drive.google.com/drive/folders/1Fh6tidH1W2aU3SbKVggg6cxWqT021rE0?usp=drive_link">Dataset</a> &nbsp&nbsp
    🏆 <a href="https://multi-trust.github.io/#leaderboard">Leaderboard</a>
</div>
<br>

<div align="center">
    <img src="https://img.shields.io/badge/Benchmark-Truthfulness-yellow" alt="Truthfulness" />
    <img src="https://img.shields.io/badge/Benchmark-Safety-red" alt="Safety" />
    <img src="https://img.shields.io/badge/Benchmark-Robustness-blue" alt="Robustness" />
    <img src="https://img.shields.io/badge/Benchmark-Fairness-orange" alt="Fairness" />
    <img src="https://img.shields.io/badge/Benchmark-Privacy-green" alt="Privacy" />
</div>
<br>

![framework](docs/structure/framework.jpg)


## Getting Started

### 💡 Environment

- Option A: Pip install
    ```shell
    conda create -n multitrust python=3.9
    conda activate multitrust

    # Note: Tsinghua Source can be discarded.
    pip install -r env/requirements.txt
    pip install xformers==0.0.26.post1 --no-deps --index-url https://pypi.tuna.tsinghua.edu.cn/simple/
    pip install flash_attn==2.5.9.post1 --index-url https://pypi.tuna.tsinghua.edu.cn/simple/
    ```

- Option B: Docker
    - (Optional) Commands to install Docker
    ```shell
    # Our docker version:
    #     Client: Docker Engine - Community
    #     Version:           27.0.0-rc.1
    #     API version:       1.46
    #     Go version:        go1.21.11
    #     OS/Arch:           linux/amd64

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit

    sudo systemctl restart docker
    sudo usermod -aG docker [your_username_here]
    ```
    - Commands to install environment
    ```shell
    #  Note: 
    # [code] is an `absolute path` of project root: abspath(./)
    # [data] and [playground] are `absolute paths` of data and model_playground(decompress our provided data/playground).
    
    docker build -t multitrust:v0.0.1 -f env/Dockerfile .

    docker run -it \
        --name multitrust \
        --gpus all \
        --privileged=true \
        --shm-size=10gb \
        -v /home/[your_user_name_here]/.cache/huggingface:/root/.cache/huggingface \
        -v /home/[your_user_name_here]/.cache/torch:/root/.cache/torch \
        -v [code]:/root/multitrust \
        -v [data]:/root/multitrust/data \
        -v [playground]:/root/multitrust/playground \
        -w /root/multitrust \
        -p 11180:22 \
        -p 8000:8000 \
        -d multitrust:v0.0.1 /bin/bash

    # entering the container by docker exec
    docker exec -it multitrust /bin/bash

    # or entering by ssh
    ssh -p 11180 root@[your_ip_here]
    ```

### :envelope: Dataset

#### License
The codebase is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.

#### Data Preparation

Refer [here](data4multitrust/README.md) for detailed instructions.

## 📚 Docs
Our document presents interface definitions for different modules and some tutorials on how to extend modules.
Running online at: https://thu-ml.github.io/MMTrustEval/

Run following command to see the docs(locally).
```shell
mkdocs serve -f env/mkdocs.yml -a 0.0.0.0:8000
```

## 📈 Reproduce results in Our paper

| ID  | Task Name                         | Metrics                                 | Type        |
|-----|-----------------------------------|-----------------------------------------|-------------|
| T.1 | Basic World Understanding         | Accuracy ($\uparrow$)                   | Dis.\&Gen.  |
| T.2 | Advanced Cognitive Inference      | Accuracy ($\uparrow$)                   | Dis.        |
| T.3 | VQA under Instruction Enhancement | Accuracy ($\uparrow$)                   | Gen.        |
| T.4 | QA under Visual Assistance        | Accuracy ($\uparrow$)                   | Gen.        |
| T.5 | Text Misleading VQA               | Accuracy ($\uparrow$)                   | Gen.        |
| T.6 | Visual Confusion VQA              | Accuracy ($\uparrow$)                   | Gen.        |
| T.7 | Visual Misleading QA              | Accuracy ($\uparrow$)                   | Dis.        |
| S.1 | Risk Identification               | Accuracy ($\uparrow$)                   | Dis.\&Gen.  |
| S.2 | Image Description                 | Toxicity Score ($\downarrow$), RtA ($\uparrow$)  | Gen.        |
| S.3 | Toxicity Content Generation       | Toxicity Score ($\downarrow$), RtA ($\uparrow$)  | Gen.        |
| S.4 | Plain Typographic Jailbreaking    | ASR ($\downarrow$), RtA ($\uparrow$)             | Gen.        |
| S.5 | Optimized Multimodal Jailbreaking | ASR ($\downarrow$), RtA ($\uparrow$)             | Gen.        |
| S.6 | Cross-modal Influence on Jailbreaking | ASR ($\downarrow$), RtA ($\uparrow$)          | Gen.        |
| R.1 | VQA for Artistic Style images     | Score ($\uparrow$)                      | Gen.        |
| R.2 | VQA for Sensor Style images       | Score ($\uparrow$)                      | Gen.        |
| R.3 | Sentiment Analysis for OOD texts  | Accuracy ($\uparrow$)                   | Dis.        |
| R.4 | Image Captioning under Untarget attack | Accuracy ($\uparrow$)               | Gen.        |
| R.5 | Image Captioning under Target attack | Attack Success Rate ($\downarrow$)    | Gen.        |
| R.6 | Textual Adversarial Attack        | Accuracy ($\uparrow$)                   | Dis.        |
| F.1 | Stereotype Content Detection      | Containing Rate ($\downarrow$)          | Gen.        |
| F.2 | Agreement on Stereotypes          | Agreement Percentage ($\downarrow$)     | Dis.        |
| F.3 | Classification of Stereotypes     | Accuracy ($\uparrow$)                   | Dis.        |
| F.4 | Stereotype Query Test             | RtA ($\uparrow$)                        | Gen.        |
| F.5 | Preference Selection in VQA       | RtA ($\uparrow$)                        | Gen.        |
| F.6 | Profession Prediction             | Pearson’s correlation ($\uparrow$)      | Gen.        |
| F.7 | Preference Selection in QA        | RtA ($\uparrow$)                        | Gen.        |
| P.1 | Visual Privacy Recognition        | Accuracy, F1 ($\uparrow$)               | Dis.        |
| P.2 | Privacy-sensitive QA Recognition  | Accuracy, F1 ($\uparrow$)               | Dis.        |
| P.3 | InfoFlow Expectation              | Pearson's Correlation ($\uparrow$)      | Gen.        |
| P.4 | PII Query with Visual Cues        | RtA ($\uparrow$)                        | Gen.        |
| P.5 | Privacy Leakage in Vision         | RtA ($\uparrow$), Accuracy ($\uparrow$) | Gen.        |
| P.6 | PII Leakage in Conversations      | RtA ($\uparrow$) | Gen.        |


Running scripts under `scripts/run` can generate the model outputs of specific tasks and corresponding primary evaluation results in either global or smaple-wise manner. Afterw that, scripts under `scrpts/score` can be used to calculate the statistical results based on the outputs and show the results reported in the paper.

### 📌 To Make Inference 

```
# bash scripts/run/*/*.sh

scripts/run
├── fairness_scripts
│   ├── f1-stereo-generation.sh
│   ├── f2-stereo-agreement.sh
│   ├── f3-stereo-classification.sh
│   ├── f3-stereo-topic-classification.sh
│   ├── f4-stereo-query.sh
│   ├── f5-vision-preference.sh
│   ├── f6-profession-pred.sh
│   └── f7-subjective-preference.sh
├── privacy_scripts
│   ├── p1-vispriv-recognition.sh
│   ├── p2-vqa-recognition-vispr.sh
│   ├── p3-infoflow.sh
│   ├── p4-pii-query.sh
│   ├── p5-visual-leakage.sh
│   └── p6-pii-leakage-in-conversation.sh
├── robustness_scripts
│   ├── r1-ood-artistic.sh
│   ├── r2-ood-sensor.sh
│   ├── r3-ood-text.sh
│   ├── r4-adversarial-untarget.sh
│   ├── r5-adversarial-target.sh
│   └── r6-adversarial-text.sh
├── safety_scripts
│   ├── s1-nsfw-image-description.sh
│   ├── s2-risk-identification.sh
│   ├── s3-toxic-content-generation.sh
│   ├── s4-typographic-jailbreaking.sh
│   ├── s5-multimodal-jailbreaking.sh
│   └── s6-crossmodal-jailbreaking.sh
└── truthfulness_scripts
    ├── t1-basic.sh
    ├── t2-advanced.sh
    ├── t3-instruction-enhancement.sh
    ├── t4-visual-assistance.sh
    ├── t5-text-misleading.sh
    ├── t6-visual-confusion.sh
    └── t7-visual-misleading.sh
```

### 📌 To Evaluate Results
```
# python scripts/score/*/*.py

scripts/score
├── fairness
│   ├── f1-stereo-generation.py
│   ├── f2-stereo-agreement.py
│   ├── f3-stereo-classification.py
│   ├── f3-stereo-topic-classification.py
│   ├── f4-stereo-query.py
│   ├── f5-vision-preference.py
│   ├── f6-profession-pred.py
│   └── f7-subjective-preference.py
├── privacy
│   ├── p1-vispriv-recognition.py
│   ├── p2-vqa-recognition-vispr.py
│   ├── p3-infoflow.py
│   ├── p4-pii-query.py
│   ├── p5-visual-leakage.py
│   └── p6-pii-leakage-in-conversation.py
├── robustness
│   ├── r1-ood_artistic.py
│   ├── r2-ood_sensor.py
│   ├── r3-ood_text.py
│   ├── r4-adversarial_untarget.py
│   ├── r5-adversarial_target.py
│   └── r6-adversarial_text.py
├── safefy
│   ├── s1-nsfw-image-description.py
│   ├── s2-risk-identification.py
│   ├── s3-toxic-content-generation.py
│   ├── s4-typographic-jailbreaking.py
│   ├── s5-multimodal-jailbreaking.py
│   └── s6-crossmodal-jailbreaking.py
└── truthfulness
    ├── t1-basic.py
    ├── t2-advanced.py
    ├── t3-instruction-enhancement.py
    ├── t4-visual-assistance.py
    ├── t5-text-misleading.py
    ├── t6-visual-confusion.py
    └── t7-visual-misleading.py
```

### 📌 Overall Results 
![result](docs/structure/overall.png)


## :black_nib: Citation
If you find our work helpful for your research, please consider citing our work.

```bibtex
@misc{zhang2024benchmarking,
      title={Benchmarking Trustworthiness of Multimodal Large Language Models: A Comprehensive Study}, 
      author={Yichi Zhang and Yao Huang and Yitong Sun and Chang Liu and Zhe Zhao and Zhengwei Fang and
              Yifan Wang and Huanran Chen and Xiao Yang and Xingxing Wei and Hang Su and Yinpeng Dong and
              Jun Zhu},
      year={2024},
      eprint={2406.07057},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }         
```
