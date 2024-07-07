if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_id=nsfw-image-description

python run_task.py --config mmte/configs/task/safety/s1-nsfw-image-description.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/safety/s1-nsfw-image-description/${model_id}/${dataset_id}.json"
