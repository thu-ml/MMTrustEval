export APIKEY_FILE='env/apikey.yml' 
source activate mllm-dev

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "d-basic-attribute"
    "d-basic-object"
    "d-basic-scene"
    "g-basic-grounding"
    "g-basic-ocr"
)

for dataset_id in "${dataset_ids[@]}";
do
    clean_dataset_id=${dataset_id#*-}
    python run_task.py --config mmte/configs/task/truthfulness/t1-${clean_dataset_id}.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/truthfulness/t1-basic/${model_id}/${dataset_id}.json"
done

