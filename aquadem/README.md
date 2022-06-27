Continuous Control with Action Quantization from Demonstrations
===
Robert Dadashi*, Leonard Hussenot*, Damien Vincent, Sertan Girgin, Anton Raichuk, Matthieu Geist, Olivier Pietquin
---

This directory contains the source code accompanying the paper:
Continuous Control with Action Quantization from Demonstrations [https://arxiv.org/abs/2110.10149](https://arxiv.org/abs/2110.10149).

# Dependencies

AQuaDem is compatible with Python 3.9. You can install the dependencies using:

    pip install -r requirements.txt

You will also need to install Mujoco (AQuaDem is compatible with Mujoco 2.1.1). Follow the install instructions [here](https://mujoco.org/download).

# Run AQuaDem

    python -m aquadem.run_aquadqn --workdir='/tmp/aquadem' --env_name='door-human-v1'
