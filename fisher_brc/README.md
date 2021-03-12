Offline Reinforcement Learning with Fisher Divergence Critic Regularization
============================================================================================================
Ilya Kostrikov, Jonathan Tompson, Rob Fergus, Ofir Nachum
-----------------------------------------------------------------------------------------

Source code to accompany [Offline Reinforcement Learning with Fisher Divergence Critic Regularization]().

If you use this code for your research, please consider citing the paper:

```
@article{kostrikov2021fbrc,
    title={Offline Reinforcement Learning with Fisher Divergence Critic Regularization},
    author={Ilya Kostrikov and Jonathan Tompson and Rob Fergus and Ofir Nachum},
    year={2021},
}
```

Install Dependencies
--------------------
```bash
pip install -m requirements.txt
```

You will also need to install Mujoco and use a valid license. Follow the install
instructions [here](https://github.com/openai/mujoco-py).


Running Training
----------------

From the root google_research directory, run:

```bash
python -m fisher_brc.train_eval_offline \
  --task_name hopper-medium-expert-v0 \
  --seed 42 \
  --alsologtostderr
```
