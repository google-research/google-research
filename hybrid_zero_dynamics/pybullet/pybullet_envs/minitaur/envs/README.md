# Simulated Minitaur Environments

This folder contains a number of simulated Minitaur environments implemented using pybullet.

The following two environments are used in the RSS paper "[Sim-to-Real: Learning Agile Locomotion For Quadruped Robots](https://arxiv.org/abs/1804.10332)":
* Galloping environment: minitaur_reactive_env.py
* Trotting  environment:  minitaur_trotting_env.py

The rest are experimental environments.

## Prerequisites
Install [TensorFlow](https://www.tensorflow.org/install/)

Install OpenAI gym
```
pip install gym
```
Install ruamel.yaml
```
pip install ruamel.yaml
```

## Examples

To run a pre-trained PPO agent that performs the galloping gait
```
python minitaur_reactive_env_example.py
```
To run a pre-trained PPO agent that performs trotting gait
```
python minitaur_trotting_env_example.py 
```

## Authors
* Jie Tan
* Tingnan Zhang
* Erwin Coumans
* Atil Iscen
* Yunfei Bai
* Danijar Hafner
* Steven Bohez
* Vincent Vanhoucke







