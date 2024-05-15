(
    # export HF_DATASETS_OFFLINE=1 
    # export TRANSFORMERS_OFFLINE=1
    # export http_proxy='127.0.0.1:7890'
    # export https_proxy='127.0.0.1:7890'
    # unset http_proxy
    # unset https_proxy

    cd /data/zhangyichi/Trustworthy-MLLM/MMTrustEval
    source activate mllm-dev
    
    model_ids=(
        gpt-4-1106-vision-preview
        gemini-pro-vision
        # claude-3-sonnet-20240229
        qwen-vl-plus
    )
    task_ids=(
        "pii-query-email-name-occupation"
        "pii-query-email-wo-name-occupation"
        "pii-query-phone-name-occupation"
        "pii-query-phone-wo-name-occupation"
        "pii-query-address-name-occupation"
        "pii-query-address-wo-name-occupation"
        "pii-query-email-name-wo-occupation"
        "pii-query-email-wo-name-wo-occupation"
        "pii-query-phone-name-wo-occupation"
        "pii-query-phone-wo-name-wo-occupation"
        "pii-query-address-name-wo-occupation"
        "pii-query-address-wo-name-wo-occupation"
    )
    
    for model_id in "${model_ids[@]}";
    do
        
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[0]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[0]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[1]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[1]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[2]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[2]}-on-${model_id}.log 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[3]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[3]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[4]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[4]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[5]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[5]}-on-${model_id}.log 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[6]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[6]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[7]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[7]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[8]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[8]}-on-${model_id}.log 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[9]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[9]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[10]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[10]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[11]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[11]}-on-${model_id}.log 2>&1 &        
        
    done

    model_ids=( 
        # "ShareGPT4V-7B"
        # "cogvlm-chat-hf"
        # "internlm-xcomposer-7b"
        # "llava-v1.5-7b"
        # "minigpt-4-vicuna-13b-v0"
        # "otter-mpt-7b-chat"
        # "qwen-vl-chat"
        # "mplug-owl2-llama2-7b"
        # "minigpt-4-llama2-7b"
        # "gpt-4-1106-vision-preview"

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
        # gpt-4-1106-vision-preview
        # gemini-pro-vision
        # # claude-3-sonnet-20240229
        # qwen-vl-plus
    )

    task_ids=(
        "pii-query-email-name-occupation"
        "pii-query-email-wo-name-occupation"
        "pii-query-phone-name-occupation"
        "pii-query-phone-wo-name-occupation"
        "pii-query-address-name-occupation"
        "pii-query-address-wo-name-occupation"
        # "pii-query-email-name-wo-occupation"
        # "pii-query-email-wo-name-wo-occupation"
        # "pii-query-phone-name-wo-occupation"
        # "pii-query-phone-wo-name-wo-occupation"
        # "pii-query-address-name-wo-occupation"
        # "pii-query-address-wo-name-wo-occupation"
    )
    
    for model_id in "${model_ids[@]}";
    do
        
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[0]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[0]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[1]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[1]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[2]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[2]}-on-${model_id}.log 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[3]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[3]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[4]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[4]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[5]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[5]}-on-${model_id}.log 2>&1 &
        wait
    done


    task_ids=(
        # "pii-query-email-name-occupation"
        # "pii-query-email-wo-name-occupation"
        # "pii-query-phone-name-occupation"
        # "pii-query-phone-wo-name-occupation"
        # "pii-query-address-name-occupation"
        # "pii-query-address-wo-name-occupation"
        "pii-query-email-name-wo-occupation"
        "pii-query-email-wo-name-wo-occupation"
        "pii-query-phone-name-wo-occupation"
        "pii-query-phone-wo-name-wo-occupation"
        "pii-query-address-name-wo-occupation"
        "pii-query-address-wo-name-wo-occupation"
    )
    
    for model_id in "${model_ids[@]}";
    do
        
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[0]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[0]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[1]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[1]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[2]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[2]}-on-${model_id}.log 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[3]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[3]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[4]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[4]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[5]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[5]}-on-${model_id}.log 2>&1 &
        wait
    done
)



# python -u test_run_new.py --task pii-query-info-sensitivity --model_id gpt-4-1106-vision-preview --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug