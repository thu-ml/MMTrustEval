
<div align="center">
    <img src="docs/structure/background.png" alt="background" style="width: 90%;"> 
</div>

<div align="center" style="font-size: 16px;">
    🌐 <a href="https://multi-trust.github.io/">Project Page</a> &nbsp&nbsp
    📖 <a href="https://arxiv.org/abs/2406.07057">arXiv Paper</a> &nbsp&nbsp
    📜 <a href="https://thu-ml.github.io/MMTrustEval/">Documentation </a> &nbsp&nbsp
    📊 <a href="https://docs.google.com/forms/d/e/1FAIpQLSd9ZXKXzqszUoLhRT5fD9ggsSZtbmYNKgFPVekSaseYU69a_Q/viewform?usp=sf_link">Dataset</a> &nbsp&nbsp
    🤗 <a href="https://huggingface.co/datasets/thu-ml/MultiTrust">Hugging Face</a> &nbsp&nbsp
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


---

> **MultiTrust** is a comprehensive benchmark designed to assess and enhance the trustworthiness of MLLMs across five key dimensions: truthfulness, safety, robustness, fairness, and privacy. It integrates a rigorous evaluation strategy involving 32 diverse tasks to expose new trustworthiness challenges. 



<div align="center">
    <img src="docs/structure/framework.jpg" alt="framework" style="width: 90%;"> 
</div>






## 🚀 News
* **`2025.03.03`** 🌟 We have released the latest results for [DeepSeek-VL2](https://huggingface.co/deepseek-ai/deepseek-vl2) on our [project website](https://multi-trust.github.io/) ！
* **`2025.02.11`** 🌟 We have released the latest results for [DeepSeek-Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B), [CogVLM2-Llama3-Chat-19B](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B) and [GLM-4v-9B](https://huggingface.co/THUDM/glm-4v-9b) on our [project website](https://multi-trust.github.io/) ！
* **`2024.11.05`** 🌟 We have released the dataset of MultiTrust on 🤗[Huggingface](https://huggingface.co/datasets/thu-ml/MultiTrust). Feel free to download and test your own model !
* **`2024.11.05`** 🌟 We have updated the toolbox to support several latest models, e.g., [Phi-3.5](https://huggingface.co/microsoft/Phi-3.5-vision-instruct), [Cambrian-13B](https://huggingface.co/nyu-visionx/cambrian-13b), [Qwen2-VL-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct), and their results have been uploaded to the [leaderboard](https://multi-trust.github.io/) !
* **`2024.09.26`** 🎉 [Our paper](https://arxiv.org/abs/2406.07057) has been accepted by the Datasets and Benchmarks track in NeurIPS 2024 ！See you in Vancouver ~
* **`2024.08.12`** 🌟 We have released the latest results for [DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL), and [hunyuan-vision](https://hunyuan.tencent.com/) on our [project website](https://multi-trust.github.io/) ！
* **`2024.07.07`** 🌟 We have released the latest results for [GPT-4o](https://openai.com/index/hello-gpt-4o/), [Claude-3.5](https://www.anthropic.com/news/claude-3-5-sonnet), and [Phi-3](https://ollama.com/library/phi3) on our [project website](https://multi-trust.github.io/) ！
* **`2024.06.07`** 🌟 We have released [MultiTrust](https://multi-trust.github.io/), the first comprehensive and unified benchmark on the trustworthiness of MLLMs !

## 🛠️ Installation

The envionment of this version has been updated to accommodate more latest models. If you want to ensure more precise replication of experimental results presented in the paper, you could switch to the branch [v0.1.0](https://github.com/thu-ml/MMTrustEval/tree/v0.1.0).

- Option A: UV install
    ```shell
    uv venv --python 3.9
    source .venv/bin/activate

    uv pip install setuptools
    uv pip install torch==2.3.0
    uv pip sync --no-build-isolation env/requirements.txt
    ```

- Option B: Docker
    - How to install docker
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
    - Get our image: 
        - B.1: Pull image from DockerHub
            ```shell
            docker pull jankinfstmrvv/multitrust:latest
            ```

        - B.2: Build from scratch
            ```shell
            #  Note: 
            # [data] is the `absolute paths` of data.

            docker build --network=host -t multitrust:latest -f env/Dockerfile .
            ```
    
    - Start a container:
        ```shell
        docker run -it \
            --name multitrust \
            --gpus all \
            --privileged=true \
            --shm-size=10gb \
            -v $HOME/.cache/huggingface:/root/.cache/huggingface \
            -v $HOME/.cache/torch:/root/.cache/torch \
            -v [data]:/root/MMTrustEval/data \
            -w /root/MMTrustEval \
            -d multitrust:latest /bin/bash

        # entering the container
        docker exec -it multitrust /bin/bash
        ```
  
- Several tasks require the use of commercial APIs for auxiliary testing. Therefore, if you want to test all tasks, please add the corresponding model API keys in [env/apikey.yml](https://github.com/thu-ml/MMTrustEval/blob/v0.1.0/env/apikey.yml).

## :envelope: Dataset

### License
- The codebase is licensed under the **CC BY-SA 4.0** license.

- MultiTrust is only used for academic research. Commercial use in any form is prohibited.

- If there is any infringement in MultiTrust, please directly raise an issue, and we will remove it immediately.

### Data Preparation

Refer [here](data4multitrust/README.md) for detailed instructions.

## 📚 Docs
Our document presents interface definitions for different modules and some tutorials on **how to extend modules**.
Running online at: https://thu-ml.github.io/MMTrustEval/

Run following command to see the docs(locally).
```shell
mkdocs serve -f env/mkdocs.yml -a 0.0.0.0:8000
```

## 📈 Reproduce results in Our paper

Running scripts under `scripts/run` can generate the model outputs of specific tasks and corresponding primary evaluation results in either a global or sample-wise manner. 
### 📌 To Make Inference 

```
# Description: Run scripts require a model_id to run inference tasks.
# Usage: bash scripts/run/*/*.sh <model_id>

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
After that, scripts under `scripts/score` can be used to calculate the statistical results based on the outputs and show the results reported in the paper.
```
# Description: Run scripts require a model_id to calculate statistical results.
# Usage: python scripts/score/*/*.py --model_id <model_id>

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
### 📌 Task List
The total 32 tasks are listed here and ○: rule-based evaluation (e.g., keywords matching); ●: automatic evaluation
by GPT-4 or other classifiers; ◐: mixture evaluation.

| ID  | Task Name                         | Metrics                                 | Task Type        | Eval       |
|-----|-----------------------------------|-----------------------------------------|-------------|--|
| T.1 | Basic World Understanding         | Accuracy ($\uparrow$)                   | Dis.\&Gen.  | ◐|
| T.2 | Advanced Cognitive Inference      | Accuracy ($\uparrow$)                   | Dis.        | ○|
| T.3 | VQA under Instruction Enhancement | Accuracy ($\uparrow$)                   | Gen.        | ●|
| T.4 | QA under Visual Assistance        | Accuracy ($\uparrow$)                   | Gen.        | ●|
| T.5 | Text Misleading VQA               | Accuracy ($\uparrow$)                   | Gen.        | ●|
| T.6 | Visual Confusion VQA              | Accuracy ($\uparrow$)                   | Gen.        | ○|
| T.7 | Visual Misleading QA              | Accuracy ($\uparrow$)                   | Dis.        | ●|
| S.1 | Risk Identification               | Accuracy ($\uparrow$)                   | Dis.\&Gen.  | ◐|
| S.2 | Image Description                 | Toxicity Score ($\downarrow$), RtA ($\uparrow$)  | Gen.        | ●|
| S.3 | Toxicity Content Generation       | Toxicity Score ($\downarrow$), RtA ($\uparrow$)  | Gen.        | ◐|
| S.4 | Plain Typographic Jailbreaking    | ASR ($\downarrow$), RtA ($\uparrow$)             | Gen.        | ◐|
| S.5 | Optimized Multimodal Jailbreaking | ASR ($\downarrow$), RtA ($\uparrow$)             | Gen.        | ◐|
| S.6 | Cross-modal Influence on Jailbreaking | ASR ($\downarrow$), RtA ($\uparrow$)          | Gen.        | ◐|
| R.1 | VQA for Artistic Style images     | Score ($\uparrow$)                      | Gen.        | ◐|
| R.2 | VQA for Sensor Style images       | Score ($\uparrow$)                      | Gen.        | ●|
| R.3 | Sentiment Analysis for OOD texts  | Accuracy ($\uparrow$)                   | Dis.        | ○|
| R.4 | Image Captioning under Untarget attack | Accuracy ($\uparrow$)               | Gen.        | ◐|
| R.5 | Image Captioning under Target attack | Attack Success Rate ($\downarrow$)    | Gen.        | ◐|
| R.6 | Textual Adversarial Attack        | Accuracy ($\uparrow$)                   | Dis.        | ○|
| F.1 | Stereotype Content Detection      | Containing Rate ($\downarrow$)          | Gen.        | ●|
| F.2 | Agreement on Stereotypes          | Agreement Percentage ($\downarrow$)     | Dis.        | ◐|
| F.3 | Classification of Stereotypes     | Accuracy ($\uparrow$)                   | Dis.        | ○|
| F.4 | Stereotype Query Test             | RtA ($\uparrow$)                        | Gen.        | ◐|
| F.5 | Preference Selection in VQA       | RtA ($\uparrow$)                        | Gen.        | ●|
| F.6 | Profession Prediction             | Pearson’s correlation ($\uparrow$)      | Gen.        | ◐|
| F.7 | Preference Selection in QA        | RtA ($\uparrow$)                        | Gen.        | ●|
| P.1 | Visual Privacy Recognition        | Accuracy, F1 ($\uparrow$)               | Dis.        | ○|
| P.2 | Privacy-sensitive QA Recognition  | Accuracy, F1 ($\uparrow$)               | Dis.        | ○|
| P.3 | InfoFlow Expectation              | Pearson's Correlation ($\uparrow$)      | Gen.        | ○|
| P.4 | PII Query with Visual Cues        | RtA ($\uparrow$)                        | Gen.        | ◐|
| P.5 | Privacy Leakage in Vision         | RtA ($\uparrow$), Accuracy ($\uparrow$) | Gen.        | ◐|
| P.6 | PII Leakage in Conversations      | RtA ($\uparrow$) | Gen.        | ◐|

### ⚛️ Overall Results 
- Proprietary models like GPT-4V and Claude3 demonstrate consistently top performance due to enhancements in alignment and safety filters compared with open-source models.
- A global analysis reveals a correlation coefficient of 0.60 between general capabilities and trustworthiness of MLLMs, indicating that more powerful general abilities could help better trustworthiness to some extent.
- Finer correlation analysis shows no significant link across different aspects of trustworthiness, highlighting the need for comprehensive aspect division and identifying gaps in achieving trustworthiness.


<div align="center">
    <img src="docs/structure/overall.png" alt="result" style="width: 90%;"> 
</div>


## :black_nib: Citation
If you find our work helpful for your research, please consider citing our work.

```bibtex
@article{zhang2024benchmarking,
  title={Benchmarking Trustworthiness of Multimodal Large Language Models: A Comprehensive Study},
  author={Zhang, Yichi and Huang, Yao and Sun, Yitong and Liu, Chang and Zhao, Zhe and Fang, Zhengwei and Wang, Yifan and Chen, Huanran and Yang, Xiao and Wei, Xingxing and others},
  journal={arXiv preprint arXiv:2406.07057},
  year={2024}
}  
```
