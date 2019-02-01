FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer "Evann Courdier <evann.courdier@epfl.ch>"

# install some necessary tools.
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    gcc \
    git \
    htop \
    iputils-ping \
    locales \
    python3.6 \
    python3-pip \
    python3.6-dev \
    python3.6-venv \
    sudo \
    tmux \
    unzip \
    vim \
    wget \
    zsh \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libboost-all-dev \
    libsdl2-dev \
&& rm -rf /var/lib/apt/lists/*

# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel numpy

# configure environments.
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# Python packages
COPY req_pip.txt install/
RUN pip3 install --upgrade -r install/req_pip.txt

# Install torch
# COPY torch-1.0.0-cp36-cp36m-linux_x86_64.whl install/
# RUN pip3 install --upgrade install/torch*.whl torchvision

# configure user.
ENV SHELL=/bin/bash \
    NB_USER=courdier \
    NB_UID=192613 \
    NB_GROUP=MLO-unit \
    NB_GID=11169
ENV HOME=/home/$NB_USER


RUN groupadd $NB_GROUP -g $NB_GID
RUN useradd -m -s /bin/bash -N -u $NB_UID -g $NB_GID $NB_USER && \
    echo "${NB_USER}:${NB_USER}" | chpasswd && \
    usermod -aG sudo,adm,root ${NB_USER}
RUN chown -R ${NB_USER}:${NB_GROUP} ${HOME}
# The user gets passwordless sudo
RUN echo "${NB_USER}   ALL = NOPASSWD: ALL" > /etc/sudoers

# Become the user and go to the home directory
USER ${NB_USER}
WORKDIR ${HOME}
