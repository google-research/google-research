# Spin-weighted spherical CNNs in JAX

This is a JAX implementation of the paper Esteves et al, "[Spin-Weighted
Spherical CNNs](https://arxiv.org/abs/2006.10731)", NeurIPS'20.

Features:

* Ability to use any combination of spin-weights per layer,
* Ability to run distributed on multiple TPUs or GPUs,
* Quantitative equivariance tests.

If you use this code, please cite the paper:

```bibtex
@inproceedings{EstevesMD20,
 author = {Esteves, Carlos and Makadia, Ameesh and Daniilidis, Kostas},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {8614--8625},
 title = {Spin-Weighted Spherical CNNs},
 volume = {33},
 year = {2020}
}
```

# Demo
The following code snippet clones the repo, installs dependencies, downloads
dataset and trains a model on spherical MNIST.

```python
ENV_DIR=~/venvs/spin_spherical_cnns
CHECKPOINT_DIR=/tmp/swscnn_training

# Clone this repo.
apt install subversion
svn export --force https://github.com/google-research/google-research/trunk/spin_spherical_cnns

# Setup virtualenv.
virtualenv $ENV_DIR
source $ENV_DIR/bin/activate

# Install dependencies.
pip install -r spin_spherical_cnns/requirements.txt

# Download and convert datasets.
# Will write to ~/tensorflow_datasets/spherical_mnist.
cd spin_spherical_cnns/spherical_mnist
tfds build
cd ../..

# Train the model.
mkdir -p $CHECKPOINT_DIR
python3 -m spin_spherical_cnns.main \
  --workdir $CHECKPOINT_DIR \
  --config spin_spherical_cnns/configs/default.py \
  --config.dataset spherical_mnist/rotated \
  --config.model_name cnn_classifier_6_layers \
  --config.combine_train_val_and_eval_on_test True
```

# Running tests
This code is extensively tested. The snippet below runs all tests.

```shell
for f in $(find spin_spherical_cnns -name *_test\.py -printf "%f\n"); do
  python3 -m spin_spherical_cnns.${f%.py}
done
```
