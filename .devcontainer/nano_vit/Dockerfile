FROM nvcr.io/nvidia/pytorch:23.01-py3

ARG USERNAME
ARG USER_UID
ARG USER_GID

ENV HOME /home/$USERNAME
ENV SHELL /bin/bash

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo curl wget git tmux python3-pip bzip2 ca-certificates \
    && apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-dev \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME  \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/pip3 /usr/bin/pip


RUN chown -R $USERNAME:$USERNAME $HOME
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --upgrade pip \
    && pip install --upgrade ipykernel ipywidgets

# # 1. Install the dependencies
RUN pip3 install git+https://github.com/openai/CLIP.git
RUN pip3 install matplotlib

# # 2. Install the NanoOWL Python package
ARG NANOOWL_REPO=NVIDIA-AI-IOT/nanoowl
ARG NANOOWL_BRANCH=main

ADD https://api.github.com/repos/${NANOOWL_REPO}/git/refs/heads/${NANOOWL_BRANCH} /tmp/nanoowl_version.json

RUN git clone --branch=${NANOOWL_BRANCH} https://github.com/${NANOOWL_REPO} nanoowl && \
    cd nanoowl && \
    git checkout main && \
    python3 setup.py develop --user

RUN pip3 install tensorrt==8.5.2.2
