# Multi-agent Minimalistic Gridworld Environment (MultiGrid)

This repo inherits and extends
[MiniGrid](https://github.com/maximecb/gym-minigrid), a simple, lightweight, and
fast gridworld environment based on the OpenAI gym API. In this version,
multiple agents can act in the environment simultaneously. Each agent has their
own partially observable view of the world. Agents may compete for resources
with each other. MultiGrid is backwards compatible with MiniGrid and can also
handle single-agent training.

## Changes from MiniGrid

Unlike MiniGrid, the MultiGrid environment is stepped by passing in an array of
actions, one for each agent. It then returns an array of observations, each in
the form of a dict (as in MiniGrid).

For more information about the observations, the environment API, and the types
of worlds available, please consult the
[MiniGrid](https://github.com/maximecb/gym-minigrid) documentation.

## Basic Usage

There is a text-based UI application which allows you to manually control each
agent to test the environment functionality:

```
python -m manual_control_multiagent.py
```

The environment being run can be selected with the `--env_name` option, eg:

```
python -m manual_control_multiagent.py --env_name MultiGrid-DoorKey-16x16-v0
```
