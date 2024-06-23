source activate mllm-dev

python run_task.py --config mmte/configs/task/fairness/f5-vision-preference.yaml --cfg-options dataset_id="vision-preference" log_file="logs/fairness/vision-preference.json"
