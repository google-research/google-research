# Deep Reinforcement Learning for Vision-Based Robotic Grasping: A Simulated Comparison of Off-Policy Methods

This codebase implements learning algorithms and experiments from [Deep
Reinforcement Learning for Vision-Based Robotic Grasping: A Simulated Comparison
of Off-Policy Methods (ICRA 2018)](https://arxiv.org/abs/1802.10264).

If you use this codebase for your research, please cite the paper:

```
@article{quillen2018deep,
  title={Deep Reinforcement Learning for Vision-Based Robotic Grasping: A Simulated Comparative Evaluation of Off-Policy Methods},
  author={Quillen, Deirdre and Jang, Eric and Nachum, Ofir and Finn, Chelsea and Ibarz, Julian and Levine, Sergey},
  journal={IEEE International Conference on Robotics and Automation},
  year={2018}
}
```

## Features

*   Several grasping environments with varying degrees of grasping difficulty.
*   Customizable DQL, MC, Supervised, Corr-MC, DDPG, PCL algorithms.
*   MC returns and elibility traces for biased returns.
*   Bash scripts for gathering data from random policies and running synchronous
    on-policy or off-policy experiments that alternate between training and
    evaluation.
*   Change one flag and it deploys to borg with high-priority chief replica
    (fast time-to-result) and multi-replica trials (to measure reproducibility).
    Experiments persisted by MLDash.
*   Scripts to run grid search over hyperparameters.

## Getting Started

The recommended way to set up these experiments is via a virtualenv

```
sudo apt-get install python-pip
python -m pip install --user virtualenv
python -m virtualenv ~/env
source ~/env/bin/activate
```

Then install the project dependencies in that virtualenv:

```
pip install -r dql_grasping/requirements.txt
```

The first step is then to collect off-policy grasping data with a random policy.

```
sh dql_grasping/run_random_collect_oss.sh
```

Then you can train with onpolicy re-collection. By default this runs Deep
Q-Learning on the `env_procedural` environment.

```
sh dql_grasping/run_train_collect_eval_oss.sh
```

