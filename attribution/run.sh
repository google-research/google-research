#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install tensorflow
pip install -r attribution/requirements.txt
python -m attribution.integrated_gradients_test
