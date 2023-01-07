# Contrastive Value Learning code
Link to paper: https://arxiv.org/abs/2211.02100

*Abstract*: Model-based reinforcement learning (RL) methods are appealing in the
offline setting because they allow an agent to reason about the consequences of
actions without interacting with the environment. Prior methods learn a 1-step
dynamics model, which predicts the next state given the current state and
action. These models do not immediately tell the agent which actions to take,
but must be integrated into a larger RL framework. Can we model the environment
dynamics in a different way, such that the learned model does directly indicate
the value of each action? In this paper, we propose Contrastive Value Learning
(CVL), which learns an implicit, multi-step model of the environment dynamics.
This model can be learned without access to reward functions, but nonetheless
can be used to directly estimate the value of each action, without requiring any
TD learning. Because this model represents the multi-step transitions
implicitly, it avoids having to predict high-dimensional observations and thus
scales to high-dimensional tasks. Our experiments demonstrate that CVL
outperforms prior offline RL methods on complex continuous control benchmarks.

If you use this repository, please consider adding the following citation:

```
@article{mazoure2022contrastive,
  title={Contrastive Value Learning: Implicit Models for Simple Offline RL},
  author={Mazoure, Bogdan and Eysenbach, Benjamin and Nachum, Ofir and Tompson, Jonathan},
  journal={Deep Reinforcement Learning Workshop, NeurIPS 2022},
  year={2022}
}
```

### Installation instructions

1. Clone the `cvl_public` repository: `svn export https://github.com/google-research/google-research/trunk/cvl_public; cd cvl_public`
2. Create an Anaconda environment: `conda create -n cvl_public python=3.9 -y`
3. Activate the environment: `conda activate cvl_public`
4. Install the dependencies: `pip install -r requirements.txt --no-deps`
5. Check that the installation worked: `chmod +x run.sh; ./run.sh`

### Running the experiments

To run the training script, please run:
```python train_acme.py -- --lp_launch_type=local_mt --debug=True```
