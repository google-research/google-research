# C-Learning: Learning to Achieve Goals via Recursive Classification

<p align="center"> Benjamin Eysenbach, &nbsp; Ruslan Salakhutinov ,&nbsp;  Sergey Levine </p>

<p align="center">
   <a href="https://arxiv.org/abs/2011.08909">paper</a>, <a href="https://ben-eysenbach.github.io/c_learning/">website</a>
</p>

**tldr**: We reframe goal-conditioned RL as the problem of predicting and controlling the future state distribution of an autonomous agent. We solve this problem indirectly by training a classifier to predict whether an observation comes from the future. Importantly, an off-policy variant of our algorithm allows us to predict the future state distribution of a new policy, without collecting new experience. While conceptually similar to Q-learning, our approach provides a theoretical justification for goal-relabeling methods employed in prior work and suggests how the goal-sampling ratio can be optimally chosen. Empirically our method outperforms these prior methods.


If you use this code, please consider adding the corresponding citation:

```
@article{eysenbach2020clearning,
  title={C-Learning: Learning to Achieve Goals via Recursive Classification},
  author={Eysenbach, Benjamin and Salakhutdinov, Ruslan and Levine, Sergey},
  journal={arXiv preprint arXiv:2011.08909},
  year={2020}
}

```

## Installation
These instructions were tested in Google Cloud Compute with Ubuntu version 18.04.


### 1. Install Mujoco
Copy your mujoco key to `~/.mujoco/mjkey.txt`, then complete the steps below:

```
sudo apt install unzip gcc libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
wget https://www.roboti.us/download/mujoco200_linux.zip -P /tmp
unzip /tmp/mujoco200_linux.zip -d ~/.mujoco
mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin" >> ~/.bashrc
```

### 2. Install Anaconda
```
wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
chmod +x Miniconda2-latest-Linux-x86_64.sh
chmod +x ./Miniconda2-latest-Linux-x86_64.sh
```
Restart your terminal so the changes take effect.


### 3. Create an Anaconda environment and install the remaining dependencies
```
conda create --name c-learning python=3.6
conda activate c-learning
pip install tensorflow==2.4.0rc0
pip install tf_agents==0.6.0
pip install gym==0.13.1
pip install mujoco-py==2.0.2.10
pip install git+https://github.com/rlworkgroup/metaworld.git@33f3b90495be99f02a61da501d7d661e6bc675c5
```

## Running Experiments

The following lines replicate the C-learning experiments on the Sawyer tasks. Training proceeds at roughly 120 FPS on an 12-core CPU machine (no GPU). The experiments in Fig. 3 ran for up to 3M time steps, corresponding to about 7 hours. Please see the discussion below for an explanation of what the various command line options do.

```
python train_eval.py --root_dir=~/c_learning/sawyer_reach --gin_bindings='train_eval.env_name="sawyer_reach"' --gin_bindings='obs_to_goal.start_index=0' --gin_bindings='obs_to_goal.end_index=3' --gin_bindings='goal_fn.relabel_next_prob=0.5' --gin_bindings='goal_fn.relabel_future_prob=0.0'

python train_eval.py --root_dir=~/c_learning/sawyer_push --gin_bindings='train_eval.env_name="sawyer_push"' --gin_bindings='train_eval.log_subset=(3, 6)' --gin_bindings='goal_fn.relabel_next_prob=0.3' --gin_bindings='goal_fn.relabel_future_prob=0.2' --gin_bindings='SawyerPush.reset.arm_goal_type="goal"' --gin_bindings='SawyerPush.reset.fix_z=True' --gin_bindings='load_sawyer_push.random_init=True' --gin_bindings='load_sawyer_push.wide_goals=True'

python train_eval.py --root_dir=~/c_learning/sawyer_drawer --gin_bindings='train_eval.env_name="sawyer_drawer"' --gin_bindings='train_eval.log_subset=(3, None)' --gin_bindings='goal_fn.relabel_next_prob=0.3' --gin_bindings='goal_fn.relabel_future_prob=0.2' --gin_bindings='SawyerDrawer.reset.arm_goal_type="goal"'

python train_eval.py --root_dir=~/c_learning/sawyer_window --gin_bindings='train_eval.env_name="sawyer_window"' --gin_bindings='train_eval.log_subset=(3, None)' --gin_bindings='SawyerWindow.reset.arm_goal_type="goal"' --gin_bindings='goal_fn.relabel_next_prob=0.5' --gin_bindings='goal_fn.relabel_future_prob=0.0'
```

Explanation of the command line arguments:

  * `train_eval.env_name`: Selects which environment to use.
  * `obs_to_goal.start_index`, `obs_to_goal.end_index`: Select a subset of the observation to use for learning the classifier and policy. This option modifies C-learning to predict and control the density $$p(s_{t+}[\text{start}:\text{end}] \mid s_t, a_t)$$. For example, the sawyer_reach task actually contains an object in coordinates 3 through 6, but we want to ignore the object position when learning reaching.
  * `goal_fn.relabel_next_prob`, `goal_fn.relabel_future_prob`: TD C-learning says that 50% of goals should be sampled from the next state distribution, corresponding to setting `goal_fn.relabel_next_prob=0.5`. The hybrid MC + TD version of C-learning described in Appendix E changes this so that some goals are also sampled from the future state distribution (corresponding to setting `goal_fn.relabel_future_prob`). For C-learning, we assume that `goal_fn.relabel_next_prob + goal_fn.relabel_future_prob == 0.5`. To prevent unintended bugs, both these parameters must be explicitly specified in the command line.
  * `*.reset.arm_goal_type`: For the Sawyer manipulation tasks, the goal state contains both the desired object position and arm position. Use `*.reset.arm_goal_type="goal"` to indicate that the arm goal position should be the same as the object goal position. Use `*.reset.arm_goal_type="puck"` to indicate that the arm goal position should be the same as the *initial* object position.
  * `load_sawyer_push.random_init`: Whether to randomize the initial arm position.
  * `load_sawyer_push.wide_goals`: Whether to sample a wider range of goals.
  
### Questions?
If you have any questions, comments, or suggestions, please reach out to Benjamin Eysenbach (eysenbach@google.com).
