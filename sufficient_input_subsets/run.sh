#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r sufficient_input_subsets/requirements.txt
python -m sufficient_input_subsets.sis_test

set +x
echo -e "\nTests/installation OK. See README.md for usage details and examples."
