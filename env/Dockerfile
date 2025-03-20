FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

COPY . /root/MMTrustEval

WORKDIR /root/MMTrustEval

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get clean all \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        apt-utils \
        wget \
        openssh-server \
        sudo \
        gcc \
        git \
        g++ \
        build-essential \
        pkg-config \
        libssl-dev \
        python3.9

RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && source $HOME/.local/bin/env \
    && uv venv --python 3.9 \
    && source .venv/bin/activate \
    && uv pip install setuptools \
    && uv pip install torch==2.3.0 \
    && uv pip sync --no-build-isolation ./env/requirements.txt
