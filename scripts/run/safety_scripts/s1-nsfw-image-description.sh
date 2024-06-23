source activate mllm-dev
python run_task.py --config mmte/configs/task/safety/s1-nsfw-image-description.yaml --cfg-options dataset_id="nsfw-image-description" log_file="logs/safety/nsfw-image-description.json"
