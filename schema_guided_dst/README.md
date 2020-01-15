# Baseline Model and Evaluation Script for DSTC8 Schema Guided Dialogue State Tracking

**Contact -** schema-guided-dst@google.com

**IMPORTANT** - Please report any issues with the code present in this
directory at the github repository for the
[dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue).

## Required packges
1. absl-py (for tests)
2. fuzzywuzzy
3. numpy
4. six
5. tensorflow


## Dataset

The dataset can be downloaded from the github repository for
[Schema Guided Dialogue State Tracking](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
challenge, which is a part of
[DSTC8](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue).

## Baseline Model

The baseline model is inspired from
[BERT-DST (Chao and Lane)](https://arxiv.org/pdf/1907.03040.pdf). Here is a
brief description of the model. Please refer to source code for more details.
We will be publishing a formal description of the model in the future. **This
model is provided "AS IS" without any warranty, express or implied. Google
disclaims all liability for any damages, direct or indirect, resulting from its
use.**

The baseline model consists of two modules:

1. **Schema Embedding Module** - A pre-trained BERT model to embed each schema
   element (intents, slots and categorical slot values) using their natural
   language description provided in the schema files in the dataset. These
   embedded representations are pre-computed and are not fine-tuned during
   optimization of model parameters in the state update module.

2. **State Update Module** - A training example is generated for each service
   frame in each user turn. We define the state update to be the difference
   between the slot values present in the current service frame and the frame
   for the same service present in the previous user utterance (if exists). This
   module is trained to predict the active intent, requested slots and state
   update using features from the current turn, the preceding system turn and
   embedded representations of schema elements for the service corresponding to
   the frame.

   For each example, the user utterance and the preceding system utterance are
   fed to a BERT model, which outputs the embedded representation and token
   level representation of the utterance pair. The baseline model doesn't
   predict all slot spans, so slot span related metrics are not reported. The
   predictions are modelled as:

   1. **Active intent** - The embedded representation of utterance pair is fused
      with each intent embedding (plus an extra embedding for NONE intent) and
      is projected to a scalar logit using a trainable projection. All intent
      logits thus obtained are normalized using softmax to yield a distribution
      over all intents for that service.

   2. **Requested slots** - A similar procedure to active intent prediction is
      followed to obtain a scalar logit for each slot by making use of the
      corresponding slot embedding. Each logit is normalized using sigmoid
      function to obtain the score for that slot. If a slot has a score > 0.5,
      it is predicted to be requested by the user.

   3. **Slot values** - Since the model only takes the last two utterances as
      input, it is trained to predict the difference in the dialogue state
      between the current user utterance and the preceding user utterance.
      During inference, the predictions are accummulated to yield the predicted
      dialogue state. The slot value updates are predicted in two stages. First,
      for each slot, a distribution of size three denoting the slot status and
      taking values NONE, DONTCARE and ACTIVE is obtained by a trainable
      projection. If the status of a slot is predicted to be NONE, its assigned
      value is assumed to be unchanged. If it is predicted to be DONTCARE, then
      a dontcare value is assigned to that slot and if it is predicted to be
      ACTIVE, then a slot value is predicted in the second stage and assigned to
      it.

      For categorical slots, prediction of slot value updates follows a similar
      procedure to active intent prediction in order to obtain a scalar logit
      for each possible value for each categorical slot. This is done by
      fusing the embedded representation of the utterance pair and the embedded
      representation of the corresponding slot value obtained in the Schema
      Embedding module. All logits for a slot are normalized using softmax to
      yield a distribution over all slot values of a categorical slot. The
      value with the maximum probability value is assiged to the slot.

      For non-categorical slots, the model is trained to identify the slot value
      update as a span in the utterance pair. For predicting spans, token level
      representations obtained from BERT are utilized. For each non-categorical
      slot, each token level representation is fused with the corresponding slot
      embedding obtained in the Schema Embedding module and is transformed into
      a scalar logit using a trainable projection. The logits for all tokens
      are normalized using softmax to obtain a distribution over all tokens.
      This distribution is trained to predict the start token index of the span.
      A similar procedure using a different set of weights is used to predict
      the distribution for the end token index of the span. During inference,
      the indices `i <= j` maximizing `start[i] + end[j]` is predicted to be the
      span boundary and the corresponding value is assigned to the slot.

### Results

Following are the preliminary results obtained using the baseline model on the
dev set of the respective datasets. We have not done much hyper-parameter tuning
for obtaining these results and the model implementation hasn't been optimized.
These numbers should be taken as indicative and may be updated in the future. In
the table below, SGD refers to the Schema-Guided Dialogue dataset. SGD-Single
model is trained and evaluated on single domain dialogues only whereas SGD-All
model has been trained and evaluated on the entire dataset. We are also
reporting results on [WOZ 2.0 dataset](https://arxiv.org/pdf/1606.03777.pdf) as
a sanity check.

| Metrics                | SGD-Single | SGD-All | WOZ 2.0 |
|------------------------|:----------:|:-------:|:-------:|
| Active Intent Accuracy | 0.966      | 0.885   | NA      |
| Requested Slots F1     | 0.965      | 0.972   | 0.970   |
| Average Goal Accuracy  | 0.776      | 0.694   | 0.915   |
| Joint Goal Accuracy    | 0.486      | 0.383   | 0.814   |

### Required files

1. `cased_L-12_H-768_A-12`: pretrained [BERT-Base, Cased] model checkpoint.
   Download link at [bert repository](https://github.com/google-research/bert).

### Training

Training comprises of three steps, all of which are done in
`train_and_predict.py`:

1. **Creating TF record file containing training examples.** Examples are
   generated for each frame present in each user utterance. The output directory
   for the dataset is specified by the flag `--dialogues_example_dir`. If this
   directory already contains the generated examples, this step is skipped.
2. **Computing embeddings of schema elements.** Embeddings are generated for
   each intent, slot and categorical slot value for each service using a
   pre-trained model. The output directory for the schema embeddings is
   specified by the flag `--schema_embedding_dir`. If this directory already
   contains the generated schema embeddings, this step is skipped.
3. **Training the model.** Model parameters are optimized and the checkpoints
   are saved in directory specified by the flag `--output_dir`.

Other flags defined in `train_and_predict.py` can be used to specify additional
options.

```shell
python -m schema_guided_dst.baseline.train_and_predict \
--bert_ckpt_dir <downloaded_bert_ckpt_dir> \
--dstc8_data_dir <downloaded_dstc8_data_dir> \
--dialogues_example_dir <output_example_dir> \
--schema_embedding_dir <output_schema_embedding_dir> \
--output_dir <output_ckpt_dir> --dataset_split train --run_mode train \
--task_name dstc8_single_domain
```

### Prediction

The prediction step generates a directory with json files containing dialogues
in a format which is identical to the one used by the dataset. The generated
json files can be used by the evaluation script. Like training, prediction
comprises of similar three steps, all of which are done in
`train_and_predict.py`. The flag `--output_dir` must be same as the one used
during training and the prediction outputs are saved inside it as a
subdirectory `pred_res_{ckpt_num}` where `ckpt_num` is the global step
corresponding to the checkpoint used to create the predictions. Multiple
checkpoints can be specified for prediction as a comma separated list using the
flag `--eval_ckpt`.


```shell
python -m schema_guided_dst.baseline.train_and_predict \
--bert_ckpt_dir <downloaded_bert_ckpt_dir> \
--dstc8_data_dir <downloaded_dstc8_data_dir> \
--schema_embedding_dir <output_schema_embedding_dir> \
--dialogues_example_dir <output_example_dir> \
--output_dir <output_dir_for_trained_model> --dataset_split dev \
--run_mode predict --task_name dstc8_single_domain \
--eval_ckpt <comma_separated_ckpt_numbers>
```

## Evaluation

Evaluation is done using `evaluate.py` which calculates the values of different
metrics defined in `metrics.py` by comparing model outputs with ground truth.
The script `evaluate.py` requires that all model predictions should be saved in
one or more json files contained in a single directory (passed as flag
`prediction_dir`). The json files must have same format as the ground truth data
provided in the challenge repository. This script can be run using the following
command:


```shell
python -m schema_guided_dst.evaluate \
--dstc8_data_dir <downloaded_dstc8_data_dir> \
--prediction_dir <generated_model_predictions> --eval_set dev \
--output_metric_file <path_to_json_for_report>
```
