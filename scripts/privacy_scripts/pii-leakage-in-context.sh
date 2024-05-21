dataset_ids=(
    "enron-email-text"
    "enron-email-image-oneinfo"
    "enron-email-image-typeinfo"
    "enron-email-image-allinfo"
    "enron-email-unrelated-image-color"
    "enron-email-unrelated-image-nature"
    "enron-email-unrelated-image-noise"
)

for dataset_id in "${dataset_ids[@]}";
do
    CUDA_VISIBLE_DEVICES=1 python run_task.py --config mmte/configs/task/pii_leakage_in_context.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/privacy/pii-leakage-in-context-${dataset_id}.json"
done