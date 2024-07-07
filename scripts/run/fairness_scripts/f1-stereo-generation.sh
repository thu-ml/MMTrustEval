if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_id=stereo-generation

python run_task.py --config mmte/configs/task/fairness/f1-stereo-generation.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/fairness/f1-stereo-generation/${model_id}/${dataset_id}.json"

