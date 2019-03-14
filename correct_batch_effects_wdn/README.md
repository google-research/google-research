# Correcting for Batch Effects Using Wasserstein Distance

This directory contains reference code for the paper
[Correcting for Batch Effects Using Wasserstein Distance](https://arxiv.org/abs/1711.00882).

The code is implemented in Tensorflow and the required packages are listed in
`requirements.txt`.

## Datasets
The datasets are two different types of embeddings derived from the raw image
dataset: https://data.broadinstitute.org/bbbc/BBBC021/. They are CellProfiler
embeddings and deep neural network embeddings.

### CellProfiler Embeddings
The original CellProfiler embeddings were downloaded from 
http://pubs.broadinstitute.org/ljosa_jbiomolscreen_2013/ as csv files.

To convert it into a dataframe and save it as an h5 file:

```
python -m correct_batch_effects_wdn.ljosa_embeddings_to_h5 \
--ljosa_data_directory=${LJOSA_DATA_DIRECTORY}
```

The h5 file would be saved at 

`${LJOSA_DATA_DIRECTORY}/ljosa_embeddings_462.h5`

We follow the paper https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3884769/ to
preprocess the CellProfiler embeddings:

```
python -m correct_batch_effects_wdn.ljosa_preprocessing \
--original_df=${LJOSA_DATA_DIRECTORY}/ljosa_embeddings_462.h5 \
--post_normalization_path=${LJOSA_DATA_DIRECTORY}/ljosa_embeddings_post_normalized.h5 \
--post_fa_path=${LJOSA_DATA_DIRECTORY}/ljosa_embeddings_post_fa.h5
```

This would generate two h5 files. The first file is at

`${LJOSA_DATA_DIRECTORY}/ljosa_embeddings_post_normalized.h5`,

where each dimension of the embeddings has been normalized by percentile
matching.

The second file is at

`${LJOSA_DATA_DIRECTORY}/ljosa_embeddings_post_fa.h5`,

where the post-normalized embeddings have been projected to embeddings with
dimension 50 by factor analysis.

### Deep Neural Network Embeddings
Deep neural network embeddings are obtained by running a pipeline on the raw
image dataset. In the pipeline, the raw images are corrected for imaging artifacts, cell patches are obtained by
cell center finding, and a pre-trained deep neural network is applied to the patch
images to obtain embeddings. Each embedding is of dimension 192, with 64 dimensions for
each of the three stains. More details
can be found in the paper https://ai.google/research/pubs/pub46293. Due to the
proprietary reason, the code and generated embeddings cannot be open sourced
here. Readers who are interested in testing the code can instead use the feature vectors
generated from [inception_v3 on TensorFlow Hub](https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1).

## Model Training
A Wasserstein distance network is trained to correct for batch effects.
### CellProfiler Embeddings

```
python -m correct_batch_effects_wdn.forgetting_nuisance \
--network_type=WassersteinNetwork \
--input_df="${LJOSA_DATA_DIRECTORY}/ljosa_embeddings_post_fa.h5" \
--num_steps_pretrain=100000 \
--num_steps=5000 \
--save_dir="${SAVE_DIR}/ljosa_embeddings_post_fa" \
--disc_steps_per_training_step=50 \
--checkpoint_interval=2000 \
--nuisance_levels=batch \
--batch_n=100 \
--target_levels=compound \
--feature_dim=50 \
--layer_width=2 \
--num_layers=2 \
--learning_rate=1e-4
```

### Deep Neural Network Embeddings

```
python -m correct_batch_effects_wdn.forgetting_nuisance \
--network_type=WassersteinNetwork \
--input_df="${LJOSA_DATA_DIRECTORY}/ljosa_deep_post_tvn.h5" \
--num_steps_pretrain=100000 \
--num_steps=5000 \
--save_dir="${SAVE_DIR}/ljosa_deep_post_tvn" \
--disc_steps_per_training_step=50 \
--checkpoint_interval=2000 \
--nuisance_levels=batch \
--batch_n=100 \
--target_levels=compound \
--feature_dim=192 \
--layer_width=2 \
--num_layers=2 \
--learning_rate=1e-4
```

## Model Evaluation
Model performance is evaluated by a number of metrics, quantifying how much 
biological signal is preserved in the embeddings and how much batch effect has
been removed after applying the learned transformation.
### CellProfiler Embeddings

```
DF_DIR="${SAVE_DIR}/ljosa_embeddings_post_fa/(('input_df', \
'ljosa_embeddings_post_fa.h5'), ('network_type', 'WassersteinNetwork'), \
('num_steps_pretrain', 100000), ('num_steps', 5000), ('batch_n', 100), \
('learning_rate', 0.0001), ('feature_dim', 50), \
('disc_steps_per_training_step', 50), ('target_levels', \
('compound',)), ('nuisance_levels', ('batch',)), ('layer_width', 2), \
('num_layers', 2), ('lambda_mean', 0.0), ('lambda_cov', 0.0), \
('cov_fix', 0.001))"

python -m correct_batch_effects_wdn.evaluate_metrics \
--transformation_file="${DF_DIR}/data.pkl" \
--input_df="${LJOSA_DATA_DIRECTORY}/ljosa_embeddings_post_fa.h5" \
--output_file="${DF_DIR}/evals.pkl" \
--num_bootstrap=200
```

### Deep Neural Network Embeddings

```
DF_DIR="${SAVE_DIR}/ljosa_deep_post_tvn/(('input_df', \
'ljosa_deep_post_tvn.h5'), ('network_type', 'WassersteinNetwork'), \
('num_steps_pretrain', 100000), ('num_steps', 5000), ('batch_n', 100), \
('learning_rate', 0.0001), ('feature_dim', 192), \
('disc_steps_per_training_step', 50), ('target_levels', ('compound',)), \
('nuisance_levels', ('batch',)), ('layer_width', 2), ('num_layers', 2), \
('lambda_mean', 0.0), ('lambda_cov', 0.0), ('cov_fix', 0.001))"

python -m correct_batch_effects_wdn.evaluate_metrics \
--transformation_file="${DF_DIR}/data.pkl" \
--input_df="${LJOSA_DATA_DIRECTORY}/ljosa_deep_post_tvn.h5" \
--output_file="${DF_DIR}/evals.pkl" \
--num_bootstrap=200
```

### Sample Code for Loading `evals.pkl`

```
import six.moves.cPickle as pickle
from tensorflow import gfile

def load_contents(file_path):
  with gfile.GFile(file_path, mode="r") as f:
    contents = f.read()
    contents = pickle.loads(contents)
  return contents

evals = load_contents(path)
```
