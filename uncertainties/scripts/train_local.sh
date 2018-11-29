#!/bin/bash

python train \
    --gin_config="${CODE_PATH}/sources/scripts/config_files/${FLAGS_algorithm}.gin" \
    --gin_config="${CODE_PATH}/sources/scripts/config_files/${FLAGS_dataset}_local.gin" \
    --workdir="$MODEL_PATH" \
    --algorithm="${FLAGS_algorithm}" \
    --dataset="${FLAGS_dataset}" --alsologtostderr

