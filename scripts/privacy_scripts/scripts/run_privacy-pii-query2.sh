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
        "ShareGPT4V-7B"
        "cogvlm-chat-hf"
        "internlm-xcomposer-7b"
        "llava-v1.5-7b"
        "minigpt-4-vicuna-13b-v0"
        "otter-mpt-7b-chat"
        "qwen-vl-chat"
        "mplug-owl2-llama2-7b"
        "minigpt-4-llama2-7b"
        # "gpt-4-1106-vision-preview"
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
        
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[0]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[0]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[1]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[1]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[2]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[2]}-on-${model_id}.log 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[3]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[3]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[4]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[4]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[5]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[5]}-on-${model_id}.log 2>&1 &
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
        
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[0]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[0]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[1]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[1]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[2]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[2]}-on-${model_id}.log 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[3]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[3]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[4]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[4]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[5]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[5]}-on-${model_id}.log 2>&1 &
        wait
    done
)



# python -u test_run_new.py --task pii-query-info-sensitivity --model_id gpt-4-1106-vision-preview --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug

(

    cd /data/zhangyichi/Trustworthy-MLLM/MMTrustEval
    source activate mllm-dev
    

    model_ids=( 
        # "ShareGPT4V-7B"
        "cogvlm-chat-hf"
        # "internlm-xcomposer-7b"
        # "llava-v1.5-7b"
        # "minigpt-4-vicuna-13b-v0"
        # "otter-mpt-7b-chat"
        # "qwen-vl-chat"
        # "mplug-owl2-llama2-7b"
        # "minigpt-4-llama2-7b"
        # "gpt-4-1106-vision-preview"
    )

    task_ids=(
        # "pii-query-email-name-occupation"
        # "pii-query-email-wo-name-occupation"
        # "pii-query-phone-name-occupation"
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
        
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[0]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[0]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[1]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[1]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[2]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[2]}-on-${model_id}.log 2>&1 &
        wait
    done

    model_ids=( 
        # "ShareGPT4V-7B"
        # "cogvlm-chat-hf"
        "internlm-xcomposer-7b"
        # "llava-v1.5-7b"
        # "minigpt-4-vicuna-13b-v0"
        # "otter-mpt-7b-chat"
        # "qwen-vl-chat"
        # "mplug-owl2-llama2-7b"
        # "minigpt-4-llama2-7b"
        # "gpt-4-1106-vision-preview"
    )

    task_ids=(
        "pii-query-email-name-occupation"
        "pii-query-email-wo-name-occupation"
        "pii-query-phone-name-occupation"
        "pii-query-phone-wo-name-occupation"
        "pii-query-address-name-occupation"
        "pii-query-address-wo-name-occupation"
    )
    
    for model_id in "${model_ids[@]}";
    do
        
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[0]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[0]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[1]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[1]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[2]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[2]}-on-${model_id}.log 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[3]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[3]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[4]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[4]}-on-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[5]} --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/pii-query-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/pii-query-${task_ids[5]}-on-${model_id}.log 2>&1 &
        wait
    done
)