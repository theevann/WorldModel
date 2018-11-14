#!/bin/bash
sudo apt-get install -y cmake zlib1g-dev libjpeg-dev libboost-all-dev gcc libsdl2-dev
conda config --append channels pytorch
conda create -y -n rl --file requirements.txt
source activate rl
pip install -r reqpip.txt
