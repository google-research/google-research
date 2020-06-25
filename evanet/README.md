# Evolving Space-Time Neural Architectures for Videos

This repository contains the code and pretrained models for EvaNet: "Evolving Space-Time Neural Architectures for Videos"
AJ Piergiovanni, Anelia Angelova, Alexander Toshev, and Michael S. Ryoo
https://arxiv.org/abs/1811.10636 publised at ICCV 2019.

This code supports inference with an ensemble of models pretrained on Kinetics-400.
An example video is included in the data directory. The video is from HMDB [1]
corresponding to a cricket activity. Running the full evaluation on the Kinetics-400 
validation set available in November 2018 (roughly 19200 videos) gives 77.2% accuracy.

To install requirements:

```bash
pip install -r evanet/requirements.txt
```

Then download the model weights and place them in data/checkpoints.

To evalute the pre-trained EvaNet ensemble on a sample video:
```bash
wget -P evanet/data/ https://storage.googleapis.com/gresearch/evanet/data/label_map.txt
wget -P evanet/data/ https://storage.googleapis.com/gresearch/evanet/data/v_CricketShot_g04_c01_flow.npy
wget -P evanet/data/ https://storage.googleapis.com/gresearch/evanet/data/v_CricketShot_g04_c01_rgb.
python -m evanet.run_evanet --checkpoints=rgb1.ckpt,rgb2.ckpt,flow1.ckpt,flow2.ckpt
```

References:
[1] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre. HMDB: A Large Video Database for Human Motion Recognition. ICCV, 2011
