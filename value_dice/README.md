Imitation Learning via Off-Policy Distribution Matching
============================================================================================================
Ilya Kostrikov, Ofir Nachum, Jonathan Tompson
-----------------------------------------------------------------------------------------

Source code to accompany [Imitation Learning via Off-Policy Distribution Matching](https://openreview.net/forum?id=Hyg-JC4FDr).

If you use this code for your research, please consider citing the paper:

```
@inproceedings{
  Kostrikov2020Imitation,
  title={Imitation Learning via Off-Policy Distribution Matching},
  author={Ilya Kostrikov and Ofir Nachum and Jonathan Tompson},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=Hyg-JC4FDr}
}
```

Install Dependencies
--------------------
```bash
pip install -m requirements.txt
```

You will also need to install Mujoco and use a valid license. Follow the install
instructions [here](https://github.com/openai/mujoco-py).

Expert Trajectories:
-------------------------------

Expert trajectories are generated using the [GAIL code](https://github.com/openai/imitation).

Running Training
----------------

From the root google_research directory, run:

```bash
python -m value_dice.train_eval \
  --expert_dir ./datasets/ \
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
