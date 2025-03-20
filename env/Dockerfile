FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variable
ENV DEBIAN_FRONTEND=noninteractive
ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV PATH=/usr/local/cuda/bin:/root/miniconda3/bin:${PATH}

# Copy the configuration file
COPY env/apt /etc/apt
COPY . /root/MMTrustEval

WORKDIR /root/MMTrustEval

SHELL ["/bin/bash", "--login", "-c"]

# System package installation and basic configuration
RUN apt-get clean all \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        apt-utils \
        rsync \
        openssh-server \
        inetutils-ping \
        wget \
        curl \
        sudo \
        git \
        zip \
        unzip \
        vim \
        libgl1 \
        gcc \
        g++ \
        build-essential \
        pkg-config \
        libssl-dev \
        ca-certificates \
        python3.9 \
    && rm -rf /var/lib/apt/lists/*

# SSH configuration
RUN /etc/init.d/ssh start \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo 'root:123456' | chpasswd \
    && service ssh restart


# UV
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && source $HOME/.local/bin/env \
    && uv venv --python 3.9 \
    && source .venv/bin/activate \
    && uv pip install setuptools \
    && uv pip install torch==2.3.0 \
    && uv pip sync --no-build-isolation ./env/requirements.txt


# Port
EXPOSE 22 8000

# SSH
CMD ["/usr/sbin/sshd", "-D"]
