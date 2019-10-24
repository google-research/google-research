# Behavior Regularized Actor Critic for Offline Reinforcement Learning.

This codebase implements learning algorithms and experiments from [Behavior Regularized Offline Reinforcement Learning](https://openreview.net/forum?id=BJg9hTNKPH).


If you use this codebase for your research, please cite the paper:

```
@article{wu2019behavior,
  title={Behavior Regularized Offline Reinforcement Learning},
  author={Wu, Yifan and Tucker, George and Nachum, Ofir},
  year={2019}
}
```

## Features

*   Behavior regularized actor critic framework for continuous control environments.
*   Obtain partially trained policies by training online.
*   Customizable data collection.
*   Customizable offline algorithmic components: different divergence for regularization, value penalty of policy regularization, Q-ensemble.
*   Pretrained behavior and cloned policies used in the paper.


## Getting Started

The recommended way to set up these experiments is via a virtualenv.

```
sudo apt-get install python-pip
python -m pip install --user virtualenv
python -m virtualenv ~/env
source ~/env/bin/activate
```

Then install the project dependencies in that virtualenv (you will need to
separately download appropriate MuJoCo files):

```
pip install -r requirements.txt
```

Augment your python path:
```
PYTHONPATH=$PYTHONPATH:/path/to/google_research/
```

Go to behavior_regularized_offline_rl/brac as your working directory.

The first step is to get partially trained policies using Soft Actor Critic (SAC).

```
python -m train_online \
--sub_dir=0 \
--env_name=HalfCheetah-v2 \
--eval_target=4000 \
--agent_name=sac \
--total_train_steps=500000 \
--gin_bindings="train_eval_online.model_params=(((300, 300), (200, 200),), 2)" \
--gin_bindings="train_eval_online.batch_size=256" \
--gin_bindings="train_eval_online.optimizers=(('adam', 0.0005),)"
```

Where eval_target is the performance threshold for saving the partially trained policy.

This will save a partially trained policy at
`$HOME/tmp/offlinerl/policies/{ENV_NAME}/sac/{SUB_DIR}/agent_partial_target`.

You may also specify --root_dir to replace '$HOME/tmp/offlinerl/policies' with other directories.

You can view training curves by launching a tensorboard on $HOME/tmp/offlinerl/policies/{ENV_NAME}/sac/{SUB_DIR} or any parent directory.


The next step is to collect data using the paritally trained policy. First, a data config file is needed (to specify which policies to use), see configs/dcfg_example.py as an example, and see policy_loader.py for more information. You can customize any data collection by writing a dcfg_{DATA_NAME}.py.

To collect data, run

```
python -m collect_data \
--sub_dir=0 \
--env_name=HalfCheetah-v2 \
--data_name=example \
--config_file=dcfg_example \
--n_samples=1000000
```

where 'example' can be replaced by any DATA_NAME.
This will save a policy at
`$HOME/tmp/offlinerl/data/{ENV_NAME}/{DATA_NAME}/{SUB_DIR}/data`.


Then you can train agents on these collected datasets by running train_offline.py.

Supported agents are
bc (behavior cloning), bcq, brac_primal, brac_dual.

For brac_primal, behavior cloning needs to be run first. Then you need to specify the save behavior policy checkpoint file to train brac_primal agents. See the following example:

```
ENV_NAME=HalfCheetah-v2
DATA_NAME=example
N_TRAIN=1000000
SUB_DIR=0
SEED=0
python -m train_offline \
--sub_dir=$B_SUB_DIR \
--env_name=$ENV_NAME \
--agent_name=bc \
--data_name=$DATA_NAME \
--total_train_steps=300000 \
--n_train=$N_TRAIN \
--seed=$SEED \
--gin_bindings="train_eval_offline.model_params=((200, 200),)" \
--gin_bindings="train_eval_offline.batch_size=256" \
--gin_bindings="train_eval_offline.optimizers=(('adam', 5e-4),)"
```

This will save a behavior policy at
`$HOME/tmp/offlinerl/learn/{ENV_NAME}/{DATA_NAME}/n{N_TRAIN}/bc/{SUB_DIR}/{SEED}`.

Then train a brac_primal agent with pretrained behavior policy

```
ALPHA=0.1
PLR=3e-04
ENV_NAME=HalfCheetah-v2
DATA=example
N_TRAIN=1000000
SEED=0
SUB_DIR=0
VALUE_PENALTY=True  # False for policy regularization
B_CKPT=$HOME/tmp/offlinerl/learn/$ENV_NAME/$DATA_NAME/n$N_TRAIN/bc/0/$SEED/agent_behavior
python -m brac.train_offline \
--sub_dir=SUB_DIR \
--env_name=$ENV_NAME \
--agent_name=brac_primal \
--data_name=$DATA \
--total_train_steps=500000 \
--gin_bindings="brac_primal_agent.Agent.behavior_ckpt_file='$B_CKPT'" \
--gin_bindings="brac_primal_agent.Agent.alpha=$ALPHA" \
--gin_bindings="brac_primal_agent.Agent.value_penalty=VALUE_PENALTY" \
--gin_bindings="train_eval_offline.model_params=(((300, 300), (200, 200),), 2)" \
--gin_bindings="train_eval_offline.batch_size=256" \
--gin_bindings="train_eval_offline.optimizers=(('adam', 1e-3), ('adam', $PLR), ('adam', 1e-3))"
```


This will save training and testing logs, checkpoints, etc, at
`$HOME/tmp/offlinerl/learn/{ENV_NAME}/{DATA_NAME}/n{N_TRAIN}/brac_primal/{SUB_DIR}/{SEED}`.

You may also specify --root_dir to replace '$HOME/tmp/offlinerl/learn' with other directories.

You can view training curves by launching a tensorboard on `$HOME/tmp/offlinerl/learn/{ENV_NAME}/{DATA_NAME}/n{N_TRAIN}/{AGENT_NAME}/{SUB_DIR}/{SEED}` or any parent directory.

For brac_primal and brac_dual, two policies are evaluated (can be seen in tensorboard): 'main' for simply taking the output from the policy network. 'max_q' for sampling multiple actions and take the max according to the learned q function.


## Generating Benchmark Data

This repository includes a number of saved policies in the trained_policies
directory. For each environment, there is a SAC-trained policy used to collect
the data, named `agent_partial_target*`. To generate the data used in the paper,
simply use the run_collect_data.sh script with the appropriate DATA_NAME in
pure, eps1, eps3, gaussian1, gaussian3.

For each of these collection schemes, there are pretrained behavior-cloned
policies based on this data. These are available in the trained_policies
subdirectories.
