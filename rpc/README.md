# Robust Predictable Control

<p align="center"> Benjamin Eysenbach, &nbsp; Ruslan Salakhutdinov, &nbsp; Sergey Levine</p>

<p align="center">
   <a href="http://arxiv.org/abs/2109.03214">paper</a>, <a href="https://ben-eysenbach.github.io/rpc/">website</a>
</p>

**Abstract**:
Many of the challenges facing today's reinforcement learning (RL) algorithms, such as robustness, generalization, transfer, and computational efficiency are closely related to compression.  Prior work has convincingly argued why minimizing information is useful in the supervised learning setting, but standard RL algorithms lack an explicit mechanism for compression. The RL setting is unique because (1) its sequential nature allows an agent to use past information to avoid looking at future observations and (2) the agent can optimize its behavior to prefer states where decision making requires few bits. We take advantage of these properties to propose a method (RPC) for learning *simple* policies. This method brings together ideas from information bottlenecks, model-based RL, and bits-back coding into a simple and theoretically-justified algorithm. Our method jointly optimizes a latent-space model and policy to be *self-consistent*, such that the policy avoids states where the model is inaccurate. We demonstrate that our method achieves much tighter compression than prior methods, achieving up to 5$\times$ higher reward than a standard information bottleneck. We also demonstrate that our method learns policies that are more robust and generalize better to new tasks.


If you use this code, please consider adding the corresponding citation:

```
@article{eysenbach2021robust,
  title={Robust Predictable Control},
  author={Eysenbach, Benjamin and Salakhutdinov, Ruslan and Levine, Sergey},
  journal={arXiv preprint arXiv:2109.03214},
  year={2021}
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
conda create --name rpc python=3.6
conda activate rpc
pip install tensorflow==2.6.0
pip install tf_agents==0.7.1
pip install git+https://github.com/openai/gym.git@9dea81b48a2e1d8f7e7a81211c0f09f627ee61a9
pip install mujoco-py==2.0.2.10
```


## Running Experiments

The following lines replicate the RPC experiments and baselines on the HalfCheetah-v2 task using a bitrate constraint of 0.3.

* RPC (ours):\
```
python train_eval.py --root_dir=~/rpc/rpc --gin_bindings='train_eval.env_name="HalfCheetah-v2"' \
  --gin_bindings='train_eval.log_prob_reward_scale="auto"' --gin_bindings='train_eval.predict_prior=True' \
  --gin_bindings='train_eval.use_recurrent_actor=False' --gin_bindings='train_eval.kl_constraint=0.3'```
  
* VIB (Igl 2019): \
```
python train_eval.py --root_dir=~/rpc/vib --gin_bindings='train_eval.env_name="HalfCheetah-v2"' \
  --gin_bindings='train_eval.log_prob_reward_scale=0.0' --gin_bindings='train_eval.predict_prior=False' \
  --gin_bindings='train_eval.use_recurrent_actor=False' --gin_bindings='train_eval.kl_constraint=0.3'
```

* VIB + reward (Lu 2020): \
```
python train_eval.py --root_dir=~/rpc/vib_reward --gin_bindings='train_eval.env_name="HalfCheetah-v2"' \
  --gin_bindings='train_eval.log_prob_reward_scale="auto"' --gin_bindings='train_eval.predict_prior=False' \
  --gin_bindings='train_eval.use_recurrent_actor=False' --gin_bindings='train_eval.kl_constraint=0.3'
```

* VIB + RNN \
```
python train_eval.py --root_dir=~/rpc/vib_rnn --gin_bindings='train_eval.env_name="HalfCheetah-v2"' \
  --gin_bindings='train_eval.log_prob_reward_scale=0.0' --gin_bindings='train_eval.predict_prior=False' \
  --gin_bindings='train_eval.use_recurrent_actor=True' --gin_bindings='train_eval.kl_constraint=0.3'
```

* State-space model \
```
python train_eval.py --root_dir=~/rpc/state_space --gin_bindings='train_eval.env_name="HalfCheetah-v2"' \
  --gin_bindings='train_eval.log_prob_reward_scale=0.0' --gin_bindings='train_eval.predict_prior=True' \
  --gin_bindings='train_eval.use_recurrent_actor=False' --gin_bindings='train_eval.kl_constraint=100000.0' \
  --gin_bindings='train_eval.actor_fc_layers=(256, 256)' --gin_bindings='train_eval.use_identity_encoder=True' \
  \\gin_bindings='train_eval.latent_dim="obs"'
```
* Latent-space model \
```
python train_eval.py --root_dir=~/rpc/latent_space --gin_bindings='train_eval.env_name="HalfCheetah-v2"' \
  --gin_bindings='train_eval.log_prob_reward_scale=0.0' --gin_bindings='train_eval.predict_prior=True' \
  --gin_bindings='train_eval.use_recurrent_actor=False' --gin_bindings='train_eval.kl_constraint=100000.0'
```

Additional notes:

* To evaluate a policy when (say) 80% of observations are missing (see Fig. 6), add the following flag:\
```--gin_bindings='train_eval.eval_dropout=(0.8,)'```


### Questions?
If you have any questions, comments, or suggestions, please reach out to Benjamin Eysenbach (eysenbach@google.com).

### References

* Lu, Xingyu, et al. "Dynamics Generalization via Information Bottleneck in Deep Reinforcement Learning." arXiv preprint arXiv:2008.00614 (2020).
* Igl, Maximilian, et al. "Generalization in Reinforcement Learning with Selective Noise Injection and Information Bottleneck." Advances in Neural Information Processing Systems 32 (2019): 13978-13990.
