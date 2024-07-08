if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "benchlmm-infrared"
    "benchlmm-lxray"
    "benchlmm-hxray"
    "benchlmm-mri"
    "benchlmm-ct"
    "benchlmm-remote"
    "benchlmm-driving"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/robustness/r2-ood-sensor.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/robustness/r2-ood-sensor/${model_id}/${dataset_id}.json"
done
