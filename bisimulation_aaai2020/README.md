# Scalable methods for computing state similarity in deterministic Markov Decision Processes

This package includes the code used to run the experiments in the paper
"Scalable methods for computing state similarity in deterministic Markov Decision Processes".

http://arxiv.org/abs/1911.09291

All of the commands below are run from the parent `google_research` directory.

It is recommended to start a virtualenv before running these commands:

```
virtualenv venv
source venv/bin/activate
```

Then install all necessary packages:

```
sudo apt install python-tk
pip install -r bisimulation_aaai2020/requirements.txt
```

## GridWorld

The code provided is a more general than what is necessary in the paper (it
supports arbitrary grid shapes and programatically constructed square grids).
We limit ourselves to describing what is necessary for reproducing the paper
results.

We used the grid file specified in `grid_world/configs/mirrored_rooms.grid`
and the gin-config file `grid_world/configs/mirrored_rooms.gin`.

To run:

```
python -m bisimulation_aaai2020.grid_world.compute_metric \
  --base_dir=/tmp/grid_world \
  --grid_file=bisimulation_aaai2020/grid_world/configs/mirrored_rooms.grid \
  --gin_files=bisimulation_aaai2020/grid_world/configs/mirrored_rooms.gin \
  --nosample_distance_pairs
```

## Atari 2600

The code provided allows you to load a trained Dopamine Rainbow agent and train
an on-policy bisimulation metric approximant on the state representations.

### Download the trained checkpoints
First we need to obtain the trained agent checkpoints. The Dopamine authors have
provided us with the trained checkpoints for the Rainbow agent on the 3 games
evaluated in Section 4.3 of their [paper](https://arxiv.org/abs/1812.06110).
They can be downloaded at the following URLs:

*  SpaceInvaders
   *  [main checkpoint file](https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/SpaceInvaders_v4/checkpoints/tf_ckpt-199.data-00000-of-00001)
   *  [index file](https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/SpaceInvaders_v4/checkpoints/tf_ckpt-199.index)
   *  [meta file](https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/SpaceInvaders_v4/checkpoints/tf_ckpt-199.meta)
*  Pong
   *  [main checkpoint file](https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/Pong_v4/checkpoints/tf_ckpt-199.data-00000-of-00001)
   *  [index file](https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/Pong_v4/checkpoints/tf_ckpt-199.index)
   *  [meta file](https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/Pong_v4/checkpoints/tf_ckpt-199.meta)
*  Asterix
   *  [main checkpoint file](https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/Asterix_v4/checkpoints/tf_ckpt-199.data-00000-of-00001)
   *  [index file](https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/Asterix_v4/checkpoints/tf_ckpt-199.index)
   *  [meta file](https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/Asterix_v4/checkpoints/tf_ckpt-199.meta)

Save these checkpoints in `/tmp/dopamine/trained_agents/${GAME}/` and
rename them to remove the `colab_samples_rainbow_SpaceInvaders_v4_checkpoints_`
prefix.

### Train the bisimulation metric approximant

```
GAME=SpaceInvaders
python -m bisimulation_aaai2020.dopamine.play \
    --base_dir=/tmp/dopamine/trained_metrics/${GAME} \
    --trained_checkpoint=/tmp/dopamine/trained_agents/${GAME}/tf_ckpt-199 \
    --gin_files=bisimulation_aaai2020/dopamine/configs/rainbow.gin \
    --gin_bindings="atari_lib.create_atari_environment.game_name='${GAME}'"
```

### Evaluate the bisimulation network (generate images and videos)
This will load the trained bisimulation metric approximant, start an
evaluation run, and report the bisimulation distances from a specified
start frame to every other frame in the episode, generating a set of
.png and .pdf files along the way. It will also try to generate a video
compiling all the .png files.

You can download the trained metrics from the paper at the following URLs:

*  SpaceInvaders
   *  [main checkpoint file](https://storage.googleapis.com/download-dopamine-rl/bisimulation_aaai2020/trained_metric_checkpoints/SpaceInvaders/tf_ckpt-4.data-00000-of-00001)
   *  [index file](https://storage.googleapis.com/download-dopamine-rl/bisimulation_aaai2020/trained_metric_checkpoints/SpaceInvaders/tf_ckpt-4.index)
   *  [meta file](https://storage.googleapis.com/download-dopamine-rl/bisimulation_aaai2020/trained_metric_checkpoints/SpaceInvaders/tf_ckpt-4.meta)
*  Pong
   *  [main checkpoint file](https://storage.googleapis.com/download-dopamine-rl/bisimulation_aaai2020/trained_metric_checkpoints/Pong/tf_ckpt-4.data-00000-of-00001)
   *  [index file](https://storage.googleapis.com/download-dopamine-rl/bisimulation_aaai2020/trained_metric_checkpoints/Pong/tf_ckpt-4.index)
   *  [meta file](https://storage.googleapis.com/download-dopamine-rl/bisimulation_aaai2020/trained_metric_checkpoints/Pong/tf_ckpt-4.meta)
*  Asterix
   *  [main checkpoint file](https://storage.googleapis.com/download-dopamine-rl/bisimulation_aaai2020/trained_metric_checkpoints/Asterix/tf_ckpt-4.data-00000-of-00001)
   *  [index file](https://storage.googleapis.com/download-dopamine-rl/bisimulation_aaai2020/trained_metric_checkpoints/Asterix/tf_ckpt-4.index)
   *  [meta file](https://storage.googleapis.com/download-dopamine-rl/bisimulation_aaai2020/trained_metric_checkpoints/Asterix/tf_ckpt-4.meta)

Save these checkpoints in `/tmp/dopamine/trained_metrics/${GAME}/checkpoints/` and
run the following command:

```
GAME=SpaceInvaders
python -m bisimulation_aaai2020.dopamine.evaluate \
  --base_dir=/tmp/dopamine/evals/${GAME} \
  --gin_bindings="atari_lib.create_atari_environment.game_name='${GAME}'" \
  --metric_checkpoint=/tmp/dopamine/trained_metrics/${GAME}/checkpoints/tf_ckpt-4
```

You can see a high-level explanatory video [here](https://youtu.be/zqjJp9FyOK0),
which includes the bisimulation metric approximants for a trained Rainbow agent
playing SpaceInvaders, Pong, and Asterix.

If you would like to cite the paper/code, please use the following BibTeX entry:

```
@inproceedings{castro20bisimulation,
  author    = {Pablo Samuel Castro},
  title     = {Scalable methods for computing state similarity in deterministic {M}arkov {D}ecision {P}rocesses},
  year      = {2020},
  booktitle = {Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20)},
}
```

