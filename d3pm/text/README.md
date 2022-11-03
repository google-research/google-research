# D3PM text experiments

This subdirectory contains the implementation of the D3PM text generation models.

This directory supports training on LM1B and text8 by default. LM1B is provided
by TFDS, but text8 must be downloaded. You can either download and unzip it
yourself in the `data/` directory (from http://mattmahoney.net/dc/text8.zip) or
let the data loader download it.

File that implements the D3PM diffusion processes: `diffusion.py`,

Files that define the training loop and experiment framework:
- `main.py` is the main executable, which instantiates all required components.
- `trainers.py` implements a generic trainable model and training loop.

Files that set up the model and data:
- `configs.py` constructs the configuration object used for the experiments.
- `model.py` implements the core transformer model used for D3PM experiments.
- `datasets.py` gives access to the text8 and LM1B datasets.

Utilities:
- `types.py` defines a number of common types and data structures.
- `utils.py` defines various other helper functions.
