source activate mllm-dev
python run_task.py --config mmte/configs/task/confaide_image.yaml --cfg-options log_file="logs/privacy/confaide_image.json"
python run_task.py --config mmte/configs/task/confaide_text_color.yaml --cfg-options log_file="logs/privacy/confaide_text_color.json"
python run_task.py --config mmte/configs/task/confaide_text.yaml --cfg-options log_file="logs/privacy/confaide_text.json"