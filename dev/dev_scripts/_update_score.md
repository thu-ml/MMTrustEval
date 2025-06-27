1. /data/zhangyichi/MMTrustEval-dev/local/_run_all_local.sh
   - slurm -N 1 --gres=gpu:1 -p FM _run_all_lcoal.sh $model_id 
2. scoring: 
   - bash scripts/score_all.sh $model_id
3. python scripts/compute_rank.py --model_id $model_id


speedup cmd
1. `bash _run_all_local_list.sh > all_task.list`
2. `bash run_runner.sh`
3. check api: 
    - `bash dev/test_api.sh`
    - `bash dev/moderation.sh`
4. check unfinished tasks: `python check_unfinished.py`
5. count availabel results: `bash test.sh`
6. `bash score.sh > score.log`
7. `python parse.py`