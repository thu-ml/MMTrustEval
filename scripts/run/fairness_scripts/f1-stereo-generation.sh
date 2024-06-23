source activate mllm-dev

python run_task.py --config mmte/configs/task/fairness/f1-stereo-generation.yaml --cfg-options dataset_id="stereo-generation" log_file="logs/fairness/stereo-generation.json"
