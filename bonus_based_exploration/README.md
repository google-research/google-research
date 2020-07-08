## Bonus based exploration

This is the code for the paper [On Bonus Based Exploration Methods In The Arcade Learning Environment](https://openreview.net/forum?id=BJewlyStDr) by Adrien Ali Taiga, Williams Fedus, Marlos C. Machado, Aaron Courville and Marc G. Bellemare (2020).

This repository currently includes the following algorithms used in the paper:
  * NoisyNetworks ([Fortunato et al. 2018](https://arxiv.org/abs/1706.10295))
  * Pseudo-counts with PixelCNN ([Ostrovski et al. 2017](https://arxiv.org/abs/1703.01310))
  * Random Network Distillation ([Burda et al. 2019](https://arxiv.org/abs/1810.12894))

### Running the code

python -m bonus_based_exploration.train \
  --gin_files=configs/dqn_rnd.gin --base_dir=/tmp/dopamine/

### Citation

You may cite us at

`
@inproceedings{Taiga2020On,
title={On Bonus Based Exploration Methods In The Arcade Learning Environment},
author={Adrien Ali Taiga and William Fedus and Marlos C. Machado and Aaron Courville and Marc G. Bellemare},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BJewlyStDr}
}
`
