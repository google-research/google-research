# This installation was confirmed on a GCP VM with one P100 GPU created with
# a Deep Learning image, specifically a Debian 10 based Deep Learning
# image with CUDA 11.3 preinstalled.

# Create the virtual environment.
conda create --name jaxbarf pip python=3.9
conda activate jaxbarf

# Install the requirements.
pip install -r requirements.txt

# Enable GPU training with JAX.
pip install -U jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
