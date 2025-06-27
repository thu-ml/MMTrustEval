
model_ids=(
    # 'phi-3.5'
    # "cambrian-8b"
    # "cambrian-13b"
    # "molmo-7b-d-0924"
    # "llava-v1.6-mistral-7b-hf"
    # "llava-v1.6-vicuna-7b-hf"
    # "llava-v1.6-vicuna-13b-hf"
    # "llama3-llava-next-8b-hf"
    # "llava-1.5-7b-hf"
    # "llava-1.5-13b-hf"
    # 'llama-3-2-chat'
    # "internvl2-8b"
    # 'mplug-owl3-7b-240728'
    # "qwen2-vl-chat"
    # "glm-4v-chat"
    # "cogvlm2-llama3-chat-19b"
    # "deepseek-janus-7b"
    "deepseek-vl2"
)

for model_id in "${model_ids[@]}"; do
    bash scripts/score_all.sh $model_id
    python scripts/compute_rank.py --model_id $model_id
done


# bash score.sh > score.log
# python parse.py