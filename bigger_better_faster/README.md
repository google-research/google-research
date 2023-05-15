# Bigger, Better, Faster: Human Level Atari with Human Level Efficiency

This repository implements the Bigger, Better, Faster (BBF) agent in JAX, building
on Dopamine. SPR [(Schwarzer et al, 2021)](spr) and SR-SPR [(D'Oro et al, 2023)](sr-spr) may also be run as hyperparameter configurations.

## Setup
To install the repository, simply run `pip install -r requirements.txt`.
Note that depending on your operating system and cuda version extra steps may be necessary to
successfully install JAX: please see [the JAX install instructions](https://pypi.org/project/jax/) for guidance.


## Training
To run a BBF agent locally, run

```
python -m bbf.train \
    --agent=BBF \
    --gin_files=bbf/configs/BBF.gin \
    --base_dir=/tmp/online_rl/bbf \
    --run_number=1
```

## References
* [Max Schwarzer, Ankesh Anand, Rishab Goel, Devon Hjelm, Aaron Courville and Philip Bachman. Data-efficient reinforcement learning with self-predictive representations. In The Ninth International Conference on Learning Representations, 2021.][spr]

* [Pierluca D'Oro, Max Schwarzer, Evgenii Nikishin, Pierre-Luc Bacon, Marc Bellemare, Aaron Courville.  Sample-efficient reinforcement learning by breaking the replay ratio barrier. In The Eleventh International Conference on Learning Representations, 2023][sr-spr]


[spr]: https://openreview.net/forum?id=uCQfPZwRaUu
[sr-spr]: https://openreview.net/forum?id=OpC-9aBBVJe
