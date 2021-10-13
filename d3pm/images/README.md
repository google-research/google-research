# D3PM Image experiments

This subdirectory contains the implementation of the D3PM image generation models.

File that implements the D3PM diffusion processes: `diffusion-categorical.py`

Files that define the training loop and experiment framework:
- `main.py` is the main executable, which instantiates all required components.
- `entry_point.py` handles setting up the training job and reading arguments.
- `gm.py` implements a generic trainable model and training loop.

Files that set up the model and data:
- `config.py` constructs the configuration object used for the experiments.
- `model.py` implements the `unet0` model in Flax.
- `datasets.py` gives access to the CIFAR 10 dataset.

Utilities:
- `checkpoints.py` helps with managing training checkpoints.
- `utils.py` defines various other helper functions.
