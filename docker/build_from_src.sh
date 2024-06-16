# docker pull --platform linux/amd64 ubuntu:22.04
# docker save -o ubuntu2204.tar 67c845845b7d

docker load < docker/ubuntu2204.tar
docker image tag 67c845845b7d ubuntu:22.04
docker image tag a22d8ccfe30c mllm:fromfile

# docker build -t mllm:fromfile -f Dockerfile.src .
docker build -t mllm:fromfile_v2 -f Dockerfile .

# docker run -it \
#     --name multitrust \
#     --gpus all \
#     --privileged=true \
#     --shm-size=10gb \
#     -v /data/zhangyichi/cache/huggingface:/root/.cache/huggingface \
#     -v /data/zhangyichi/MMTrustEval-dev/framework:/root/MMTrustEval-dev/framework \
#     -v /data/zhangyichi/tmp:/root/tmp \
#     -v /data/zhangyichi/Trustworthy-MLLM/data:/root/MMTrustEval-dev/framework/data \
#     -v /data/zhangyichi/Trustworthy-MLLM/playground:/root/MMTrustEval-dev/framework/playground \
#     -p 11181:22 \
#     -d mllm:fromfile /bin/bash

docker run -it \
    --name fromfile \
    --gpus all \
    --privileged=true \
    --shm-size=10gb \
    -v /data/zhangyichi/cache/huggingface:/root/.cache/huggingface \
    -v /data/home/zhangyichi/.cache/torch:/root/.cache/torch \
    -v /data/zhangyichi/MMTrustEval-dev/framework:/root/MMTrustEval-dev/framework \
    -v /data/zhangyichi/tmp:/root/tmp \
    -v /data/zhangyichi/Trustworthy-MLLM/data:/root/MMTrustEval-dev/framework/data \
    -v /data/zhangyichi/Trustworthy-MLLM/playground:/root/MMTrustEval-dev/framework/playground \
    -w /root/MMTrustEval-dev/framework \
    -p 11180:22 \
    -d mllm:fromfile /bin/bash

docker exec -it multitrust /bin/bash
docker exec -it env_test /bin/bash
docker exec -it fromfile /bin/bash

docker cp /data/zhangyichi/MMTrustEval-dev/framework/docker/cuda_12.1.1_530.30.02_linux.run env_test:/root


echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
echo 'root:123456' | chpasswd
service ssh restart

ssh -p 11180 root@10.254.31.6





pip install -r requirements.txt  --no-deps

pipreqs --ignore docker/,data/,logs/,results/,playground/ --force --use-local --savepath req.txt ./