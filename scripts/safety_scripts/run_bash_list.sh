export FLAGS_fraction_of_gpu_memory_to_use=0.9

model_name_list=('minigpt-4-vicuna-13b-v0'
                 )

GPU_ID=1
log_prefix=/data/zhangyichi/Trustworthy-MLLM/output/safety
task_id=jailbreak_safebench
num=${#model_name_list[@]}

for id in `seq 0 $((num-1))`
    do
        echo ${id}
        echo ${model_name_list[$id]}
    done

# gray box model is ResNeXt50, so we only apply EoT on ResNeXt50_32x4d model
# --model_id mplug-owl-llama-7b --output_dir /data/zhangyichi/Trustworthy-MLLM/output/safety/$TASK_ID
for idx in `seq 0 $((num-1))`
    do
        wait
        CUDA_VISIBLE_DEVICES=${GPU_ID} python test_run_george.py \
            --task=${task_id} \
            --model_id=${model_name_list[$idx]} \
            --output_dir=${log_prefix}/${task_id} &
    done

