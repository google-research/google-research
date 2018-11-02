#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install tf-nightly
pip install tfp-nightly
pip install -r probabilistic_vqvae/requirements.txt
python -m probabilistic_vqvae.mnist_experiments \
          --max_steps=500 \
          --latent_size 2 --num_codes 2 --code_size 4 --base_depth 2 --batch_size 4 \
          --nouse_autoregressive_prior
