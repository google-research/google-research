#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r graph_embedding/watch_your_step/requirements.txt

curl http://sami.haija.org/graph/datasets.tgz > datasets.tgz
tar zxvf datasets.tgz
export DATA_DIR=datasets

# note -- these are not the recommended settings for this dataset.  This is just so the open-source tests will finish quickly.
python -m graph_embedding.watch_your_step.graph_attention_learning --dataset_dir ${DATA_DIR}/wiki-vote --transition_powers 2 --max_number_of_steps 10
