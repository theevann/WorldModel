#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

sudo apt-get install -y cmake zlib1g-dev libjpeg-dev libboost-all-dev gcc libsdl2-dev
conda config --append channels pytorch
conda env create -n rl --file $DIR/env.yml
#conda create -y -n rl --file $DIR/req_conda.txt
source activate rl || { echo "ABORTING" && exit 1; }
#pip install -r $DIR/req_pip.txt
