# docker build -t mllm:fromfile -f Dockerfile.src .

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV http_proxy=http://u-LDSRbU:Pq56RrRZ@10.255.128.108:3128 \
    https_proxy=http://u-LDSRbU:Pq56RrRZ@10.255.128.108:3128
ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV PATH=/root/miniconda3/bin:${PATH}

WORKDIR /root/
COPY ./sources.list /etc/apt/sources.list
# COPY . /root/framework/
COPY ./docker/cuda_12.1.1_530.30.02_linux.run /root/cuda.run
COPY ./requirements.txt /root/requirements.txt

SHELL ["/bin/bash", "--login", "-c"]


RUN apt-get clean all \
    && apt-get update -y \
    && apt-get install -y apt-utils rsync openssh-server inetutils-ping wget curl sudo git zip unzip vim \
    && apt-get install -y libgl1 gcc g++ build-essential pkg-config libssl-dev \
    && /etc/init.d/ssh start \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo 'root:123456' | chpasswd \
    && service ssh restart \
    && mkdir -p ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm -rf ~/miniconda3/miniconda.sh \
    && bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)" \
    && /root/miniconda3/bin/conda init bash \
    && bash -l -c "source /root/.bashrc" \
    && conda install python=3.9 -y \
    # && wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run -O /root/cuda.run \
    && bash /root/cuda.run --silent --toolkit --override \
    && rm -f /root/cuda.run

RUN pip install -r /root/requirements.txt \
    && pip install xformers==0.0.26.post1 --no-deps --index-url  https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip install flash_attn==2.5.9.post1 --index-url  https://pypi.tuna.tsinghua.edu.cn/simple/


EXPOSE 22