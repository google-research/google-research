#!/bin/bash
set -e
set -x

mkdir -p pretrained_models
cd pretrained_models/

ID=1uQm8hBLB_1BA5GWKyHU2QP7wKl5vVkLr
gdown --id $ID
