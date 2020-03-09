# Revisiting Fundamentals of Experience Replay
This is the code for the paper `Revisiting Fundamentals of Experience Replay` by
William Fedus, Prajit Ramachandran, Rishabh Agarwal, Yoshua Bengio, Hugo
Larochelle, Mark Rowland and Will Dabney

### Setup
All of the commands below are run from the parent `google_research` directory.
Start a virtualenv with these commands:

```
virtualenv -p python3 .
source ./bin/activate
```

Then install necessary packages:

```
pip install -r experience_replay/requirements.txt
```

## Running the Code
To train the agent execute,

```
python -m experience_replay.train \
  --gin_files=experience_replay/configs/dqn.gin \
  --schedule=continuous_train_and_eval \
  --base_dir=/tmp/experience_replay \
  --gin_bindings=experience_replay.replay_memory.prioritized_replay_buffer.WrappedPrioritizedReplayBuffer.replay_capacity=1000000 \
  --gin_bindings=ElephantDQNAgent.oldest_policy_in_buffer=250000 \
  --gin_bindings="ElephantDQNAgent.replay_scheme='uniform'" \
  --gin_bindings="atari_lib.create_atari_environment.game_name='Pong'"
```

These correspond to the default hyperparameters. The replay ratio may be 
adjusted by changing the `oldest_policy_in_buffer`.
