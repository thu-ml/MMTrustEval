source activate mllm-dev
categories=(
    "cartoon"
    "handmake"
    "painting"
    "sketch"
    "tattoo"
    "weather"
)

for category in "${categories[@]}";
do
    CUDA_VISIBLE_DEVICES=1 python run_task.py --config mmte/configs/task/robustness/r1-ood-artistic.yaml --cfg-options dataset_id="coco-o-${category}" log_file="logs/robustness/ood-artistic-${category}.json"
done
