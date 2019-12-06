Imitation Learning via Off-Policy Distribution Matching
============================================================================================================
Ilya Kostrikov, Ofir Nachum, Jonathan Tompson
-----------------------------------------------------------------------------------------

Source code to accompany Imitation Learning via Off-Policy Distribution Matching.

Install Dependencies
--------------------
```bash
pip install -m requirements.txt
```

You will also need to install Mujoco and use a valid license. Follow the install
instructions [here](https://github.com/openai/mujoco-py).

Downloading Expert Trajectories:
-------------------------------

Comming soon.

Running Training
----------------

From the root google_research directory, run:

```bash
python -m value_dice.train_eval \
  --expert_dir ./experts/ \
  --save_dir ./save/ \
  --algo value_dice \
  --env_name HalfCheetah-v2 \
  --seed 42 \
  --num_trajectories 1 \
  --alsologtostderr
```

To reproduce results run:

```bash
sh value_dice/run_experiments.sh
```
