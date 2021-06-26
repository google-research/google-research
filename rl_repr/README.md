# RL Repr: Contrastive Representation Learning for Reinforcement (and Imitation) Learning.

This repository contains code for
* [Representation Matters: Offline Pretraining for Sequential Decision Making](https://arxiv.org/abs/2102.05815)
* [Provable Representation Learning for Imitation with Contrastive Fourier Features](https://arxiv.org/abs/2105.12272)

Please cite these work upon using this library.

## Contrastive Fourier

### Setup

First create data from the decision tree environment:

    git clone git@github.com:google-research/dice_rl.git
    pip install -e ./dice_rl
    python dice_rl/scripts/create_dataset.py --save_dir=$HOME/tmp/ --env_name=lowrank_tree --num_trajectory=500 --max_trajectory_length=3 --alpha=-0.125
    python dice_rl/scripts/create_dataset.py --save_dir=$HOME/tmp/ --env_name=lowrank_tree --num_trajectory=5 --max_trajectory_length=3 --alpha=1.125
    
### Run contrastive fourier

    cd contrastive_fourier

Run contrastive learning with fourier features:
    
    python run_tabular_bc.py --load_dir=$HOME/tmp --max_trajectory_length=3 --num_trajectory=500 --num_expert_trajectory=5 --env_name=lowrank_tree --embed_dim=16 --alpha=-0.125 --alpha_expert=1.125 --embed_learner=energy

Run vanilla BC without representation learning:

    python run_tabular_bc.py --load_dir=$HOME/tmp --max_trajectory_length=3 --num_trajectory=500 --num_expert_trajectory=5 --env_name=lowrank_tree --alpha=-0.125 --alpha_expert=1.125 --embed_learner=sgd --latent_policy=0 --finetune=1

## Attentive Contrastive Learning

### Setup

Navigate to the root of project, and perform:

    pip install -e .

Note d4rl requires [mujoco](https://github.com/openai/mujoco-py) to be properly setup.


    cd batch_rl

### Run ACL
Imitation learning:

    python train_eval_offline.py -- --algo_name=bc --downstream_mode=offline --embed_learner=acl --state_embed_dim=256 --embed_training_window=8 --embed_pretraining_steps=200000 --task_name=ant-medium-v0 --downstream_task_name=ant-expert-v0 --downstream_data_size=10000 --proportion_downstream_data=0.1

Offline RL:

    python train_eval_offline.py -- --algo_name=brac --downstream_mode=offline --embed_learner=acl --state_embed_dim=256 --embed_training_window=8 --embed_pretraining_steps=200000 --task_name=ant-expert-v0

Online RL:

    python train_eval_offline.py -- --algo_name=sac --downstream_mode=online --embed_learner=acl --state_embed_dim=256 --embed_training_window=8 --embed_pretraining_steps=200000 --task_name=ant-expert-v0 --state_mask_value=zero --state_mask_eval=1 --state_mask_dims=1 --state_mask_index=random

