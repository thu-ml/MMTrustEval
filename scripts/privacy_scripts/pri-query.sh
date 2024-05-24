source activate mllm-dev
python run_task.py --config mmte/configs/task/pri-query-vispr.yaml --cfg-options log_file="logs/privacy/pri-query-vispr.json"
python run_task.py --config mmte/configs/task/pri-query-vizwiz-drawline.yaml --cfg-options log_file="logs/privacy/pri-query-vizwiz-drawline.json"
python run_task.py --config mmte/configs/task/pri-query-vizwiz.yaml --cfg-options log_file="logs/privacy/pri-query-vizwiz.json"
