
# sync remote:
#   ssh ln01
#   cd /data/zhangyichi/MMTrustEval-dev/framework
#   git pull mmtrusteval

branch_name=dev-new-models
worktree_dir=../$branch_name
git branch $branch_name
git worktree add $worktree_dir $branch_name
# git worktree remove $worktree_dir --force

cp env/apikey_local.yml $worktree_dir/env/apikey_local.yml
cp _run_all_local.sh $worktree_dir/_run_all_local.sh
cp _worktree.sh $worktree_dir/_worktree.sh
ln -s /data/zhangyichi/Trustworthy-MLLM/playground $worktree_dir/playground
ln -s /data/zhangyichi/Trustworthy-MLLM/data/ $worktree_dir/data