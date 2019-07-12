This directory contains Python script to run linear dynamical system clustering
experiments locally without dependencies on Google packages.

## Installation

The files in this directory are meant to be run as python scripts. To install
dependencies, run `pip install -r requirements.txt` If there are error messages
related to the pylds package, do the following: 1. Install numpy through `pip
install numpy` 2. Install Cython through `pip install Cython` 3. Remove pylds
from the requirements file, and install the rest requirements `pip install -r
requirements.txt` 4. Install pylds through `pip install pylds`

## Example Usage

Example to run experiments for learning eigenvalues: python
experiment_learn_eig.py --output_dir=may11_eig_2d_test/ \
--min_seq_len=20 --max_seq_len=1000 --num_sampled_seq_len=2 \
--num_repeat=1 --hidden_dim=2 --true_eig=0.1,0.8

Example to run experiments for running clustering experiments: python
experiments.py --output_dir=may19_3d --hidden_state_dim=3 \
--min_seq_len=100 --max_seq_len=2000 --num_sampled_seq_len=20 \
--num_systems=100 --num_repeat=100 \
--cluster_center_dist_lower_bound=0.1 --hide_inputs=true

For more information about the flags, please see documentation in the scripts.
