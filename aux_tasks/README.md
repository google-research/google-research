# Aux Tasks

## Installation

To run experiments in this directory:

`git clone https://github.com/google-research/google_research.git`

`cd google_research`

We recommend doing the following in a virtual environment. (E.g. `python3 -m venv .venv && source .venv/bin/activate`)

`pip install --upgrade pip`

`pip install -r aux_tasks/requirements.txt`

If running experiments with atari, install atari roms:

`pip install gym[accept-rom-license]`

## Running Experiments

The following sections give examples of how to run each of the
experiments included in this project.

### auxiliary_mc

```
python -m aux_tasks.auxiliary_mc.train \
  --agent_name=cumulant_jax_dqn \
  --gin_files='aux_tasks/auxiliary_mc/dqn.gin' \
  --base_dir=/tmp/online_rl/dqn \
  --gin_bindings="Runner.evaluation_steps=10" \
  --gin_bindings="JaxDQNAgent.min_replay_history = 40" \
  --gin_bindings="Runner.max_steps_per_episode = 10" \
  --gin_bindings="OutOfGraphReplayBufferWithMC.replay_capacity = 10000" \
  --gin_bindings="OutOfGraphReplayBufferWithMC.batch_size = 5" \
  --gin_bindings="Runner.training_steps = 100"
```

### grid

The grid experiments use a distributed learning setup, consisting of
multiple workers, a replay buffer, and learner.

To run a worker:

```
python -m aux_tasks.grid.actor \
  --config=aux_tasks/grid/config.py:implicit \
  --reverb_address=localhost:1234 \
  --eval=False
```

Note that you will need to run at least one train worker (`--eval=False`) and
at least one eval worker (`--eval=True`).

To start the replay buffer:

```
python -m aux_tasks.grid.server \
  --port=1234 \
  --config=aux_tasks/grid/config.py:implicit
```

To start the learner:

```
python -m aux_tasks.grid.train \
  --base_dir=/tmp/pw \
  --reverb_address=localhost:1234 \
  --config=aux_tasks/grid/config.py:implicit
```

### minigrid

```
python -m aux_tasks.minigrid.train \
  --env_name=classic_fourrooms \
  --base_dir=/tmp/minigrid
```

### synthetic

```
python -m aux_tasks.synthetic.run_synthetic
```
