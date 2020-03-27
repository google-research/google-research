#!/bin/bash
set -e
set -x

virtualenv -p python3 --system-site-packages tmpenv
source tmpenv/bin/activate
pip3 install -r requirements.txt

mkdir /tmp/ieg

CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_0.4 \
--network_name=resnet29 --checkpoint_path=/tmp/ieg

