#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

sudo apt-get install -y cmake zlib1g-dev libjpeg-dev libboost-all-dev gcc libsdl2-dev
conda config --append channels pytorch
conda create -y -n rl --file $DIR/requirements.txt
source activate rl || { echo "ABORTING" && exit 1; }
pip install -r $DIR/reqpip.txt
