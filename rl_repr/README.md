# RL Repr: Contrastive Representation Learning for Reinforcement (and Imitation) Learning.

This repository contains code for
* [TRAIL: Near-Optimal Imitation Learning with Suboptimal Data](https://openreview.net/pdf?id=6q_2b6u0BnJ)
* [Provable Representation Learning for Imitation with Contrastive Fourier Features](https://arxiv.org/abs/2105.12272)
* [Representation Matters: Offline Pretraining for Sequential Decision Making](https://arxiv.org/abs/2102.05815)

Please cite these work upon using this library.

## TRAIL: Transition-Reparametrized Actions for Imitation Learning

### Setup

Navigate to the root of project, and run:

    pip install -e .

Note d4rl requires [mujoco](https://github.com/openai/mujoco-py) to be properly setup.


### Run TRAIL
Run BC with TRAIL:

    source scripts/run_trail_bc.sh
    

## State Representation with Contrastive Fourier

### Setup

First create data from the decision tree environment:

    git clone git@github.com:google-research/dice_rl.git
    pip install -e ./dice_rl
    python dice_rl/scripts/create_dataset.py --save_dir=$HOME/tmp/ --env_name=lowrank_tree --num_trajectory=500 --max_trajectory_length=3 --alpha=-0.125
    python dice_rl/scripts/create_dataset.py --save_dir=$HOME/tmp/ --env_name=lowrank_tree --num_trajectory=5 --max_trajectory_length=3 --alpha=1.125
    cd contrastive_fourier

### Run contrastive fourier

Run contrastive learning with fourier features:
    
    python run_tabular_bc.py --load_dir=$HOME/tmp --max_trajectory_length=3 --num_trajectory=500 --num_expert_trajectory=5 --env_name=lowrank_tree --embed_dim=16 --alpha=-0.125 --alpha_expert=1.125 --embed_learner=energy

Run vanilla BC without representation learning:

    python run_tabular_bc.py --load_dir=$HOME/tmp --max_trajectory_length=3 --num_trajectory=500 --num_expert_trajectory=5 --env_name=lowrank_tree --alpha=-0.125 --alpha_expert=1.125 --embed_learner=sgd --latent_policy=0 --finetune=1

## Attentive Contrastive Learning

### Setup

Navigate to the root of project, and run:

    pip install -e .

Note d4rl requires [mujoco](https://github.com/openai/mujoco-py) to be properly setup.


### Run ACL
Imitation learning:

    source scripts/run_acl_bc.sh

Offline RL:

    source scripts/run_acl_brac.sh

Online RL:

    source scripts/run_acl_sac.sh

