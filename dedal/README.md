# Dedal: Deep embedding and alignment of protein sequences.
This package contains all the necessary tools to reproduce the experiments presented in the [dedal paper](https://www.biorxiv.org/content/10.1101/2021.11.15.468653v2).

## Install
To install dedal, it is necessary to clone the google-research repo:
```
git clone https://github.com/google-research/google-research.git
```
From the `google_research` folder, you may install the necessary requirements by executing:
```
pip install -r dedal/requirements.txt
```
## Pre-trained model

DEDAL is available in [TensorFlow Hub](https://tfhub.dev/google/dedal/3).

The model expects a `tf.Tensor<tf.int32>[2B, 512]` as inputs, representing a batch of B sequence pairs to be aligned right-padded to a maximum length of 512, including the special EOS token. Pairs are expected to be arranged consecutively in this batch, that is, `inputs[2*b]` and `inputs[2*b + 1]` represent the b-th sequence pair with b ranging from 0 up to B - 1 (inclusive).

By default, the model runs in "alignment" mode and its output is a Python dict containing:
+ 'sw_scores': a `tf.Tensor<tf.float32>[B]` with the alignment scores
+ 'homology_logits': a `tf.Tensor<tf.float32>[B]` with the homology detection logits
+ 'paths': a `tf.Tensor<tf.float32>[B, 512, 512, 9]` representing the predicted alignments
+ 'sw_params': a tuple of three `tf.Tensor<tf.float32>[B, 512, 512]` containing the contextual Smith-Waterman parameters (substitution scores, gap open and gap extend penalties)

Additional signatures are provided to run the model in "embedding" mode, in which case it returns a single `tf.Tensor<tf.float32>[2B, 512, 768]` with the embeddings of each input sequence.

```python
import tensorflow as tf
import tensorflow_hub as hub
from dedal import infer  # Requires google_research/google-research.

dedal_model = hub.load('https://tfhub.dev/google/dedal/3')

# "Gorilla" and "Mallard" sequences from [1, Figure 3].
protein_a = 'SVCCRDYVRYRLPLRVVKHFYWTSDSCPRPGVVLLTFRDKEICADPRVPWVKMILNKL'
protein_b = 'VKCKCSRKGPKIRFSNVRKLEIKPRYPFCVEEMIIVTLWTRVRGEQQHCLNPKRQNTVRLLKWY'
# Represents sequences as `tf.Tensor<tf.float32>[2, 512]` batch of tokens.
inputs = infer.preprocess(protein_a, protein_b)

# Aligns `protein_a` to `protein_b`.
align_out = dedal_model(inputs)

# Retrieves per-position embeddings of both sequences.
embeddings = dedal_model.call(inputs, embeddings_only=True)

# Postprocesses output and displays alignment.
output = infer.expand(
    [align_out['sw_scores'], align_out['paths'], align_out['sw_params']])
output = infer.postprocess(output, len(protein_a), len(protein_b))
alignment = infer.Alignment(protein_a, protein_b, *output)
print(alignment)

# Displays the raw Smith-Waterman score and the homology detection logits.
print('Smith-Waterman score (uncorrected):', align_out['sw_scores'].numpy())
print('Homology detection logits:', align_out['homology_logits'].numpy())
```

## Data

DEDAL was trained and evaluated on publicly available data, namely, the March
2018 release of UniRef50 and Pfam 34.0. The subdirectory `preprocessing` contains
all of the code necessary to reproduce the preprocessing pipeline described in
the paper. Sequence identifiers corresponding to each Pfam-A seed split
can be downloaded [here](https://drive.google.com/file/d/11S2OdnduXM3id7F3k6kUxi8_qXJC8bav/view?usp=sharing).

## Training from scratch

The repository also contains all of the code necessary to reproduce the different
training phases from scratch. For now, we assume the preprocessing pipeline
has been run locally on the raw, publicly available data before training. We aim
to make the preprocessed data available in the following weeks to simplify this
process.

For example, to train a DEDAL model from scratch on the masked language modelling,
pairwise sequence alignment and homology detection tasks, one may run the following
command from the `google_research` folder:
```
python3 -m dedal.main -- \
  --base_dir /tmp/dedal \
  --gin_config data/uniref50.gin \
  --gin_config data/pfam34_alignment.gin \
  --gin_config data/pfam34_homology.gin \
  --gin_config model/dedal.gin \
  --gin_config task/finetune.gin \
  --gin_bindings UNIREF50_DATA_DIR=/path/to/masked_lm/data \
  --gin_bindings PFAM34_ALIGNMENT_DATA_DIR=/path/to/alignment/data \
  --gin_bindings PFAM34_HOMOLOGY_DATA_DIR=/path/to/homology/data \
  --gin_bindings MAIN_VOCAB_PATH=/path/to/main/vocab \
  --gin_bindings TOKEN_REPLACE_VOCAB_PATH=/path/to/token_replace/vocab \
  --gin_bindings ALIGNMENT_PATH_VOCAB_PATH=/path/to/alignment_path/vocab \
  --task train \
  --alsologtostderr
```
Note that the transformer architecture might be slow to train on a CPU and running on accelerators would greatly improve the training speed. The pre-trained DEDAL model was fit using 32 TPU v3 cores.

The `base_dir` flag is the folder where to write the checkpoints and to log the metrics. In the example above, it would be `/tmp/dedal`. To visualize the logged metrics, one can simply start a TensorBoard pointing to the given folder, such as:
```
%tensorboard --logdir /tmp/dedal
```

In case the training is interrupted, restarting the same command would not start the training over from scratch, but from the last available checkpoint. The frequency of checkpointing and logging can be changed from the `task` Gin-config files, e.g., `task/finetune.gin`.

The `task` flag enables to either run a training, an evaluation of a downstream training with its own eval. In evaluation mode, the training checkpoints will be loaded on the fly until the last one has been reached, such that one can run
both an eval process along with a training one, so that the evaluation does not slow the training down. Alternatively, one can set `separate_eval=False` in the training loop so that eval and train will be run alternatively.

## License
Licensed under the Apache 2.0 License.

## Disclaimer
This is not an official Google product.
