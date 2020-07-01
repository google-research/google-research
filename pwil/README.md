PWIL: Primal Wasserstein Imitation Learning
===
Robert Dadashi, Leonard Hussenot, Matthieu Geist, Olivier Pietquin
---

This directory contains the source code accompanying the paper:
Primal Wasserstein Imitation Learning [https://arxiv.org/abs/2006.04678](https://arxiv.org/abs/2006.04678).

# Dependencies

PWIL is compatible with Python 3.7.7. You can install the dependencies using:

    pip install -r requirements.txt

You will also need to install Mujoco and use a valid license. Follow the install
instructions [here](https://github.com/openai/mujoco-py).

# Expert demonstrations
We are working on making expert demonstrations available.

# Run PWIL

    python -m pwil.trainer --workdir='/tmp/pwil' --env_name='Hopper-v2' --demo_dir=$DEMO_DIR

