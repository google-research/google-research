# Training Text-to-Text Transformers with Privacy Guarantees

Natalia Ponomareva, Jasmijn Bastings, Sergei Vassilvitskii

## Abstract

Recent advances in NLP often stem from large transformer-based pre-trained models, which rapidly grow in size and use more and more training data. Such models are often released to the public so that end users can fine-tune them on a task dataset. While it is common to treat pre-training data as public, it may still contain personally identifiable information (PII), such as names, phone numbers, and copyrighted material. Recent findings show that the capacity of these models allows them to memorize parts of the training data, and suggest differentially private (DP) training as a potential mitigation. While there is recent work on DP fine-tuning of NLP models, the effects of DP pre-training are less well understood: it is not clear how downstream performance is affected by DP pre-training, and whether DP pre-training mitigates some of the memorization concerns. We focus on T5 and show that by using recent advances in JAX and XLA we can train models with DP that do not suffer a large drop in pre-training utility, nor in training speed, and can still be fine-tuned to high accuracies on downstream tasks (e.g. GLUE). Moreover, we show that T5’s span corruption is a good defense against data memorization.

## Paper

[https://aclanthology.org/2022.findings-acl.171/](https://aclanthology.org/2022.findings-acl.171/)


## Code

This code depends on the following libraries that need to be installed first:

- [T5X](https://github.com/google-research/t5x)
- [T5] (https://github.com/google-research/text-to-text-transfer-transformer)
- [SeqIO](https://github.com/google/seqio)
- [Flaxformer](https://github.com/google/flaxformer)

To include them, you can create a submodule for those repositories. See more
on submodules here https://github.blog/2016-02-01-working-with-submodules/.

## Notice

This is not an official Google product.


## Commands

### Installation
```sh

git clone https://github.com/google/private_text_transformers /tmp/private_text_transformers
git submodule add https://github.com/google-research/t5x
```

Also please follow https://github.com/google-research/t5x#installation to make
sure t5x and dependencies (including xmanager) are properly installed.


### Re-training a SentencePiece
For some of the experiments in the paper, you might want to retrain a SentencePiece
tokenizer (with or without differential privacy).

Please use an implementation of SentencePiece from https://github.com/google/sentencepiece
and set the other parameters accordingly.

To retrain a sentence piece with privacy:

```sh
spm_train -- \
   --input=<input> \
   --input_format="tsv" \
   ...
   --enable_differential_privacy="true" \
   --differential_privacy_noise_level=<noise_level> \
   --differential_privacy_clipping_threshold=<clipping_threshold>
```

### Training models

#### Pretrain a model

A script below shows how to

- Provide a custom sentence piece to the model. If you don't need to override
SentencePiece, do not include gin.seqio.SentencePieceVocabulary.sentencepiece_model_file and
gin.seqio.SentencePieceVocabulary.extra_ids

- Train with differential privacy (DP-Train). Setting --gin.USE_DP=False
reverts to a default non-DP training. 

```sh
# Export GOOGLE_CLOUD_BUCKET_NAME to a proper value.
export GOOGLE_CLOUD_BUCKET_NAME=...
export MODEL_DIR=gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x_retrieval/$(date +%Y%m%d)

python ./t5x/scripts/xm_launch.py \
--gin_file="t5x/t5x/examples/t5/t5_1_0/small.gin" \
--gin_file="private_text_transformers/private_t5x/configs/custom_pretrain.gin" \
--gin.TRAIN_STEPS=100000 \
--gin.MIXTURE_OR_TASK_NAME="'c4_v220_span_corruption_custom_sp'" \
--gin.MIXTURE_OR_TASK_MODULE="'private_text_transformers.private_t5x.tasks'" \
--gin.TASK_FEATURE_LENGTHS="{'inputs': 512, 'targets': 114}" \
--name=baseline \
--gin.seqio.SentencePieceVocabulary.sentencepiece_model_file="'YOUR_SENTENCE_PIECE_FILE'" \
--gin.seqio.SentencePieceVocabulary.extra_ids=100 \
--gin.USE_CACHED_TASKS=False \
--gin.USE_DP=True --gin.DP_L2_CLIP_NORM=1. --gin.DP_NOISE_MULTIPLIER=0.5 \
--gin.LOSS_NORMALIZING_FACTOR="'NUM_REAL_TARGET_TOKENS'" \
--gin.train/DatasetConfig.batch_size=4096 --gin.train_eval/DatasetConfig.batch_size=4096 \
--gin.MODEL_DIR=\"${MODEL_DIR}\" 
 
``` 
  
#### Finetune on GLUE tasks

```sh
python ./t5x/scripts/xm_launch.py \
--gin_file=private_text_transformers/private_t5x/finetune/t5_custom_sentencepiece_small_glue.gin \
--gin.MIXTURE_OR_TASK_MODULE="'private_text_transformers.private_t5x.tasks'" \
--gin.seqio.SentencePieceVocabulary.sentencepiece_model_file="'YOUR_SENTENCE_PIECE_FILE'" \
--gin.seqio.SentencePieceVocabulary.extra_ids=100 \
--gin.TRAIN_STEPS=250000 \
--gin.USE_CACHED_TASKS=False \
--gin.INITIAL_CHECKPOINT_PATH="'PRETRAINED_CHECKPOINT'" \
--name=finetuned_on_glue \ 
--gin.MODEL_DIR=\"${MODEL_DIR}\"

```

### Testing memorization
We use scripts from https://github.com/google-research/deduplicate-text-datasets
to create Train dup, Train unique, Valid in train and Valid unique subsets of
c4 data.
After you created these datasets, add a mixture in tasks.py as follows

```sh
_INFERENCE_PROMPTS_DIR = "gs://WHERE_YOUR_DATA_IS"


_INFERENCE_PROMPTS_FEATURE_DESCRIPTION = {
    "inputs": tf.io.FixedLenFeature([], dtype=tf.string),
    "targets": tf.io.FixedLenFeature([], dtype=tf.string),
}

for inference_prompt_type in [
    "train_dup",
    "train_unique",
    "valid_in_train",
    "valid_unique",
]:

  TaskRegistry.add(
      inference_prompt_type,
      source=seqio.TFExampleDataSource(
          split_to_filepattern={
              "validation":
                  os.path.join(_INFERENCE_PROMPTS_DIR,
                               f"{inference_prompt_type}.tfrecord*")
          },
          feature_description=_INFERENCE_PROMPTS_FEATURE_DESCRIPTION),
      output_features=get_custom_output_features(),
      preprocessors={
          seqio.preprocessors.tokenize_and_append_eos,
      },
      metric_fns=[
          t5_metrics.accuracy, t5_metrics.edit_distance,
          t5_metrics.sequence_accuracy
      ])

```
 
Once the mixture was added, you can evaluate on this mixture as follows.

```sh
INFER_OUTPUT_DIR="..."  # directory to write infer output
T5X_DIR="..."  # directory where the t5x is cloned, e.g., ${HOME}"/t5x".
MIXTURE="train_dup" # Say on which mixture to infer.
CHECKPOINT_PATH="..."

python3 ${T5X_DIR}/t5x/infer.py \
  --gin_file="t5x/t5x/examples/t5/t5_1_0/small.gin" \
  --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file="'YOUR_SENTENCE_PIECE_FILE'" \
  --gin.seqio.SentencePieceVocabulary.extra_ids=100 \
  --gin.MIXTURE_OR_TASK_NAME=\'${MIXTURE}\' \
  --gin.MIXTURE_OR_TASK_MODULE="'private_text_transformers.private_t5x.tasks'" \
  --gin.TASK_FEATURE_LENGTHS=\{\'inputs\':\ 512,\ \'targets\':\ 114\} --gin.DatasetConfig.split=\'validation\'
  --gin.CHECKPOINT_PATH=\"${CHECKPOINT_PATH}\" 
  --gin.INFER_OUTPUT_DIR=\"${INFER_OUTPUT_DIR}\" 
 

```  

