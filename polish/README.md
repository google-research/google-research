# Policy Optimization by Local Improvement through Search (POLISH)

This repo contains code for the paper Policy Optimization by Local Improvement through Search.

The codebase is branched from the Tree Search Policy Optimization codebase from Amir Yazdanbakhsh (ayazdan@google.com), with help from the team. The major modification made by Jialin Song (jialins@google.com) is how the MCTS rollouts are performed and the training loss during policy updates.

## Prerequisites
### Gin https://github.com/google/gin-config
### OpenAI Gym: https://github.com/openai/gym
### MuJoCo https://github.com/openai/mujoco-py

## Usage

From the directory where the polish project is downloaded, run:
```
bash ./run.sh
```
