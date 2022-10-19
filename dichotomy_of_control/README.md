# Dichotomy of Control: Separating What You Can Control from What You Cannot

This repository contains code for ``Dichotomy of Control: Separating What You Can Control from What You Cannot''.

Please cite this work upon using this library.

## Setup

Install the Bandit and FrozenLake environments:

    git clone git@github.com:google-research/dice_rl.git
    pip install -e ./dice_rl

Install the remaining dependencies:

    pip install -e . --use-deprecated=legacy-resolver
    
## Create offline datasets

Create the offline datasets for Bandit and FrozenLake:

    source create_dataset.sh

## Run Dichotomy of Control

Run DoC on Bandit:

    python -m dichotomy_of_control.scripts.run_tabular.py --load_dir='./tests/testdata' --algo_name='doc'

Change `algo_name` in `run_tabular.sh` from `doc` (Dichotomy of Control) to `dt` (Decision Transformer) to observe dramatic performance drop.


Run DoC on FrozenLake:

    python -m dichotomy_of_control.scripts.run_neural_dt --load_dir='./tests/testdata'

