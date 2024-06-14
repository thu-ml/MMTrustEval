# docker build -t multitrust:v1.0 ./docker

FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace
COPY ./requirements.txt /workspace/
COPY ./sources.list /etc/apt/sources.list

SHELL ["/bin/bash", "-c"]

RUN apt-get clean all \
    && apt-get update -y \
    && apt-get install -y libgl1 \
    && apt-get install -y apt-utils rsync openssh-server inetutils-ping wget curl sudo git zip unzip vim \
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
    && source /root/.bashrc

EXPOSE 22
EXPOSE 7890