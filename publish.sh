#!/bin/bash
cp req_pip.txt docker/req_pip.txt
cd docker && docker build . -t courdier-ae
docker tag courdier-ae ic-registry.epfl.ch/mlo/courdier_experiment
docker push ic-registry.epfl.ch/mlo/courdier_experiment
