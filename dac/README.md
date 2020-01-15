Discriminator-Actor-Critic: Addressing Sample Inefficiency and Reward Bias in Adversarial Imitation Learning
============================================================================================================
Ilya Kostrikov, Kumar Krishna Agrawal, Debidatta Dwibedi, Sergey Levine, Jonathan Tompson
-----------------------------------------------------------------------------------------

Source code to accompany our [paper](https://arxiv.org/abs/1809.02925).

Install Dependencies
--------------------

We use Python 3.5.4rc1. You may also need to install a number of dependencies.

    pip3 install gym
    pip3 install --upgrade tensorflow tensorflow_probability
    pip3 install absl-py

You will also need to install Mujoco and use a valid license. Follow the install
instructions [here](https://github.com/openai/mujoco-py).

Generating / Downloading Expert Trajectories:
-------------------------------

Clone the repo of expert trajectories:

    cd /data/dac/  # We will assume access to this directory.
    git clone git@github.com:ikostrikov/gail-experts.git

Then use our import script to turn them into checkpoints (~1-2 hours):

    python3 generate_expert_data.py \
      --src_data_dir /data/dac/gail-experts/ \
      --dst_data_dir /data/dac/gail-experts/

Running Training
----------------

Launch run_training_worker.sh to start the training worker. Then in another
terminal, launch run_evaluation_worker.sh. Training takes approximately 1 to 2
hours.

To change the environment, number of expert trajectories, etc, edit the
variables defined in the bash scripts above.

To see reward results live during training, launch a tensorboard:

    tensorboard --logdir /tmp/lfd_state_action_traj_4_HalfCheetah-v2_20
