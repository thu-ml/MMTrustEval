(
    # export HF_DATASETS_OFFLINE=1 
    # export TRANSFORMERS_OFFLINE=1

    # unset http_proxy
    # unset https_proxy

    cd /data/zhangyichi/Trustworthy-MLLM/MMTrustEval
    source activate mllm-dev
        

    shot=zeroshot
    echo "shot-mode:$shot"

    task_ids=(
        pii-leakage-in-context-text
        pii-leakage-in-context-images
        pii-leakage-in-context-unrelated-images-nature
        pii-leakage-in-context-unrelated-images-noise 
        pii-leakage-in-context-unrelated-images-color 
    )    

    model_ids=( 
        minigpt-4-llama2-7b
        minigpt-4-vicuna-13b-v0
        llava-v1.5-7b
        llava-v1.5-13b
        ShareGPT4V-13B
        llava-rlhf-13b
        LVIS-Instruct4V
        otter-mpt-7b-chat
        internlm-xcomposer-7b
        internlm-xcomposer2-vl-7b
        mplug-owl-llama-7b
        mplug-owl2-llama2-7b
        InternVL-Chat-ViT-6B-Vicuna-13B
        qwen-vl-chat
        cogvlm-chat-hf
        llava-v1.6-13b
        instructblip-flan-t5-xxl
    )


    for model_id in "${model_ids[@]}";
    do
        CUDA_VISIBLE_DEVICES=3 nohup python -u test_run_zw.py --task ${task_ids[0]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-leakage-in-context-${shot}-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/${task_ids[0]}-${shot}-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=4 nohup python -u test_run_zw.py --task ${task_ids[1]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-leakage-in-context-${shot}-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/${task_ids[1]}-${shot}-${model_id}.log 2>&1 &
        wait
    done
    for model_id in "${model_ids[@]}";
    do
        CUDA_VISIBLE_DEVICES=3 nohup python -u test_run_zw.py --task ${task_ids[2]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-leakage-in-context-${shot}-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/${task_ids[2]}-${shot}-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=4 nohup python -u test_run_zw.py --task ${task_ids[3]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-leakage-in-context-${shot}-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/${task_ids[3]}-${shot}-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[4]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-leakage-in-context-${shot}-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/${task_ids[4]}-${shot}-${model_id}.log 2>&1 &
        wait
    done

)
