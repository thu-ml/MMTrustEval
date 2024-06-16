# install docker on server
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo systemctl restart docker
sudo usermod -aG docker zhangyichi

# save and load docker image
docker save -o ubuntu_docker_image.tar e4c58958181a
docker load < docker/ubuntu_docker_image.tar
docker image tag d583c3ac45fd ubuntu:latest
docker image tag f626b3781841 mllm:apt_dep


# run a container
docker run --gpus all -p 11181:22 --privileged=true --shm-size=10gb -itd ubuntu:latest /bin/bash
docker exec -it fc4e85ef3deb /bin/bash


docker build -t multitrust:v1.0 ./docker
docker run --gpus all -p 11181:22 --privileged=true --shm-size=10gb -it f626b3781841 /bin/bash
docker exec -it b7c4eb0e0dbde64d95ea42b4b85d9d3f899bb374159db2a7881c81480dcadfde /bin/bash


docker run --gpus all -p 11180:22 --privileged=true --shm-size=10gb --name multitrust -itd mllm:apt_dep /bin/bash
docker exec -it multitrust /bin/bash

docker run -it \
    --name multitrust \
    --gpus all \
    --privileged=true \
    --shm-size=10gb \
    -v /data/home/zhangyichi/.cache:/root/.cache \
    -v /data/zhangyichi/cache/huggingface:/root/.cache/huggingface \
    -v /data/zhangyichi/MMTrustEval-dev/framework:/workspace/MMTrustEval-dev/framework \
    -v /data/zhangyichi/tmp:/workspace/tmp \
    -p 11180:22 \
    -d mllm:apt_dep /bin/bash

docker exec -it multitrust /bin/bash



echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
echo 'root:123456' | chpasswd
service ssh restart

ssh -p 11180 root@10.254.31.6




docker export -o multitrust.tar 133262345ea9


docker commit -a "jankinf" -m "mllm-dev env" 133262345ea9  mllm:conda_dep
docker commit -a "jankinf" -m "mllm-dev env" af79020a95c9  mllm:all_dep









docker run -it \
    --name multitrust \
    --gpus all \
    --privileged=true \
    --shm-size=10gb \
    -v /data/home/zhangyichi/.cache:/root/.cache \
    -v /data/zhangyichi/cache/huggingface:/root/.cache/huggingface \
    -v /data/zhangyichi/MMTrustEval-dev/framework:/workspace/MMTrustEval-dev/framework \
    -v /data/zhangyichi/tmp:/workspace/tmp \
    -v /data/zhangyichi/Trustworthy-MLLM/data:/workspace/MMTrustEval-dev/framework/data \
    -v /data/zhangyichi/Trustworthy-MLLM/playground:/workspace/MMTrustEval-dev/framework/playground \
    -p 11181:22 \
    -d mllm:all_dep /bin/bash

docker exec -it multitrust /bin/bash

pipreqs --ignore docker/,data/,logs/,results/,playground/ --force --use-local --savepath req.txt ./