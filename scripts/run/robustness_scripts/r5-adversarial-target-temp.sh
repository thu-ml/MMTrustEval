if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_id=adv-target

CUDA_VISIBLE_DEVICES=1 python run_task_temp.py --config mmte/configs/task/robustness/r5-adversarial-target.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/robustness/r5-adversarial-target/${model_id}/${dataset_id}.json"
