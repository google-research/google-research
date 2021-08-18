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
sudo apt install subversion
svn export --force https://github.com/google-research/google-research/trunk/spin_spherical_cnns

# Setup virtualenv.
python3 -m venv $ENV_DIR
source $ENV_DIR/bin/activate

# Install dependencies.
pip install -r spin_spherical_cnns/requirements.txt
# For training with GPUs, you might have to change this to your
# system's CUDA version.
pip install --upgrade "jax[cuda110]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

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
  --config.model_name spin_classifier_6_layers \
  --config.combine_train_val_and_eval_on_test True
```

Running the code above on a [GCP Deep Learning
VM](https://cloud.google.com/deep-learning-vm), with 1 x NVIDIA Tesla P100, 16
vCPUs and 60 GB memory achieves a training speed of around 12.4 steps/s and final
accuracy of 99.37% on rotated Spherical MNIST. Total training and
evaluation time is about 35 min. See sample output below.

```
I0817 20:43:06.907417 140619461363520 train.py:309] num_train_steps=22500, steps_per_epoch=1875
I0817 20:47:43.610097 140588926080768 logging_writer.py:57] Hyperparameters: {'checkpoint_every_steps': 1000, 'combine_train_val_and_eval_on_test': True, 'dataset': 'spherical_mnist/rotated', 'eval_every_steps': 1000, 'eval_pad_last_batch': True, 'learning_rate': 0.001, 'learning_rate_schedule': 'cosine', 'log_loss_every_steps': 100, 'model_name': 'spin_classifier_6_layers', 'num_epochs': 12, 'num_eval_steps': -1, 'num_train_steps': -1, 'per_device_batch_size': 32, 'seed': 42, 'shuffle_buffer_size': 1000, 'trial': 0, 'warmup_epochs': 1, 'weight_decay': 0.0}
I0817 20:47:43.612029 140619461363520 train.py:358] Starting training loop at step 1.
I0817 20:47:43.612589 140588926080768 logging_writer.py:35] [1] param_count=39146
I0817 20:47:43.677301 140619461363520 train.py:162] train_step(batch={'input': Traced<ShapedArray(float32[32,64,64,1,1])>with<DynamicJaxprTrace(level=0/1)>, 'label': Traced<ShapedArray(int32[32])>with<DynamicJaxprTrace(level=0/1)>})
I0817 20:47:43.678994 140619461363520 train.py:128] get_learning_rate(step=Traced<ShapedArray(int32[])>with<DynamicJaxprTrace(level=0/1)>, base_learning_rate=0.001, steps_per_epoch=1875, num_epochs=12)
I0817 20:48:14.027691 140619461363520 train.py:383] Finished training step 1.
I0817 20:48:14.229008 140619461363520 train.py:383] Finished training step 2.
I0817 20:48:14.244100 140619461363520 train.py:383] Finished training step 3.
I0817 20:48:14.317119 140619461363520 train.py:383] Finished training step 4.
I0817 20:48:14.397786 140619461363520 train.py:383] Finished training step 5.
I0817 20:48:27.649730 140588926080768 logging_writer.py:35] [100] learning_rate=5.333333683665842e-05, loss=2.4577670097351074, loss_std=0.13251596689224243, train_accuracy=0.10062500089406967
I0817 20:48:35.714306 140588926080768 logging_writer.py:35] [200] learning_rate=0.00010666667367331684, loss=2.03875470161438, loss_std=0.14517973363399506, train_accuracy=0.26749998331069946
I0817 20:48:43.778739 140588926080768 logging_writer.py:35] [300] learning_rate=0.00016000001050997525, loss=1.7513136863708496, loss_std=0.12867669761180878, train_accuracy=0.3646875023841858

(...)

I0817 21:18:18.211259 140588926080768 logging_writer.py:35] [22300] learning_rate=2.3198128928925144e-07, loss=0.007606287021189928, loss_std=0.016141731292009354, train_accuracy=0.9975000023841858
I0817 21:18:26.271346 140588926080768 logging_writer.py:35] [22400] learning_rate=5.799532232231286e-08, loss=0.008844759315252304, loss_std=0.018938612192869186, train_accuracy=0.9975000023841858
I0817 21:18:34.252626 140588926080768 logging_writer.py:35] [22500] steps_per_sec=12.409270
I0817 21:18:34.351791 140619461363520 train.py:268] Starting evaluation.
I0817 21:18:34.415858 140619461363520 train.py:225] eval_step(batch={'input': Traced<ShapedArray(float32[32,64,64,1,1])>with<DynamicJaxprTrace(level=0/1)>, 'label': Traced<ShapedArray(int32[32])>with<DynamicJaxprTrace(level=0/1)>, 'mask': Traced<ShapedArray(bool[32])>with<DynamicJaxprTrace(level=0/1)>})
I0817 21:18:53.154831 140588926080768 logging_writer.py:35] [22500] accuracy=0.9937000274658203, eval_loss=0.055803120136260986
I0817 21:18:53.155782 140619461363520 train.py:410] Finishing training at step 22500
```

# Running tests
This code is extensively tested. The snippet below runs all tests.

```shell
for f in $(find spin_spherical_cnns -name *_test\.py -printf "%f\n"); do
  python3 -m spin_spherical_cnns.${f%.py}
done
```
