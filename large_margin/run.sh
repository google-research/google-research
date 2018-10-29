#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install tensorflow
pip install -r large_margin/requirements.txt
python -m large_margin.margin_loss_test
