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

We currently only support running T5X from Cloud TPU VMs, which are in alpha
phase. Cloud TPU VMs currently require a special jaxlib build.

If you are an alpha user, you should have received instructions on how to
install the right version of jaxlib.

Next, you should clone this repository and install the requirements. Below are
all instructions.

```
# Install SVN to only download the T5X directory of Google Research.
sudo apt install subversion
svn export https://github.com/google-research/google-research/trunk/flax_models/t5x


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

# Disable C++ jitting. This is due to a mismatch between `jax` and `jaxlib` and
# will soon be fixed.
echo "export JAX_CPP_JIT=0" >> ~/.bashrc && . ~/.bashrc
```

## Running train

The following command fine-tunes a p5x-small model on the GLUE tasks and store
the results in the folder `p5x_data`.

```
python3 -m p5x.train --config=p5x/configs/t5_small_glue.py --model_dir=p5x_data
```
