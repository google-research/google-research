#! /bin/bash
set -e
set -x
virtualenv -p python3 .
source ./bin/activate

pip3 install tensorflow torch jax jaxlib
pip3 install git+https://github.com/google-research/flax.git@prerelease

cd opt_list/

python3 -m opt_list.examples.tf_v1
python3 -m opt_list.examples.tf_keras
python3 -m opt_list.examples.torch
python3 -m opt_list.examples.jax_flax
python3 -m opt_list.examples.jax_optimizers
python3 -m opt_list.examples.jax_optix
