# T5X: T5 + JAX + Flax

**WARNING:** This codebase is *highly experimental* and will be replaced with a
more robust open-source release soon. Please do not rely on it if you need a
stable codebase. We are currently not accepting Pull Requests.

## JAX on Cloud TPU

This code is not intended to work with the "JAX Cloud TPU Preview" that uses
tpu_driver, but rather only with the "new JAX on Cloud TPU in private alpha"
that involves direct SSH access. In order to sign up for private alpha access,
please [following this link](http://goo.gle/jax-tpu-signup).

If you running into issues, please post them on the "JAX on Cloud TPU" chatroom
that you have access to through the alpha signup.

## Installation

Please run the following code directly from your VM terminal with Cloud TPUs.

We first set some environment variables correctly and store them in `.bashrc`.

```
# Disable C++ jitting. This is due to a mismatch between `jax` and `jaxlib` and
# will soon be fixed.
echo "export JAX_CPP_JIT=0" >> ~/.bashrc

# Include this folder in your PATH, which is not by default on the VM and gives
# some warnings.
echo "PATH=$PATH:/home/$USER/.local/bin" >> ~/.bashrc

# Source bashrc so changes are applied to this terminal as well.
. ~/.bashrc
```

Then run the following commands to install all dependencies.

```
# Install SVN to only download the T5X directory of Google Research.
sudo apt install subversion
svn export https://github.com/google-research/google-research/trunk/flax_models/t5x

# Upgrade pip.
pip install --user --upgrade pip

# Install the requirements from `requirements.txt`.
pip install --user -r t5x/requirements.txt

# Install a special version of `libtpu` is required.
/usr/bin/docker-credential-gcr configure-docker
sudo docker rm libtpu
sudo docker create --name libtpu gcr.io/cloud-tpu-v2-images/libtpu:libtpu_20201102_RC00 "/bin/bash"
sudo docker cp libtpu:libtpu.so /lib

# Install a version of `jaxlib` that supports TPU.
PYTHON_VERSION=cp36  # Supported python versions: cp36, cp37, cp38
pip install --upgrade --user "https://storage.googleapis.com/jax-releases/tpu/jaxlib-0.1.58+tpu20201129-${PYTHON_VERSION}-none-manylinux2010_x86_64.whl"
```

## Running train

The following command fine-tunes a T5X-small model on the GLUE tasks and store
the results in the folder `t5x_data`.

```
python3 -m t5x.train --config=t5x/configs/t5_small_glue.py --model_dir=t5x_data
```
