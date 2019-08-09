## BERT Baseline for FAT-NQ

This repo contains code for training and evaluating a BERT baseline model on the
Natural Questions using Wikidata Background knowledge.

### Requirements

Please install sling from : https://github.com/google/sling

### Data Preparation

Assuming we have an entity linked version of NQ. The training set for the
Natural Question is quite large so we precompute all the features for it as
tensorflow examples. Precomputation can be performed with:

```
python -m fat.fat_bert_nq.prepare_nq_data \
  --logtostderr \
  --input_jsonl ~/data/nq-train-??.jsonl.gz \
  --output_tfrecord ~/output_dir/nq-train.tfrecords-00000-of-00001 \
  --max_seq_length=512 \
  --include_unknowns=0.02 \
  --vocab_file=bert-joint-baseline/vocab-nq.txt
```

This operation is quite slow, so we normally run this in parallel on all the NQ
training shards and then shuffle the output in a single file. You can skip this
command by simply using the preprocessed training set above.

### Evaluating Our Pretrained Model

You can then evaluate our model on this data with:

```
python -m fat.fat_bert_nq.run_nq \
  --logtostderr \
  --bert_config_file=bert-joint-baseline/bert_config.json \
  --vocab_file=bert-joint-baseline/vocab-nq.txt \
  --predict_file=tiny-dev/nq-dev-sample.no-annot.jsonl.gz \
  --init_checkpoint=bert-joint-baseline/bert_joint.ckpt \
  --do_predict \
  --output_dir=bert_model_output

python -m natural_questions.nq_eval \
  --logtostderr \
  --gold_path=tiny-dev/nq-dev-sample.jsonl.gz \
  --predictions_path=bert_model_output/predictions.json
```

A full evaluation on the NQ development set can be performed by pointing the
above commands to the full dev set `nq-dev-0?.jsonl.gz` through the
`--predict_file` and `--gold_path` flags. Note however that this evaluation is
also rather slow since it runs a large BERT model on all the text in the dev
set. It should complete in approximately 4 hours on a Tesla P100.

### Training Our Model

Assuming you have access to a TPU, you should be able to train a model
comparable to ours with the following command in a few hours, although you might
need to further tune the learning rate between 1e-4 and 1e-5 and the number of
train epochs between 1 and 3. In our paper we initialize our training from a
BERT model trained on SQuAD2.0 and then finetune on NQ for only 1 epoch with a
learning rate of 3e-5.

```
python -m fat.fat_bert_nq.run_nq \
  --logtostderr \
  --bert_config_file=bert-joint-baseline/bert_config.json \
  --vocab_file=bert-joint-baseline/vocab-nq.txt \
  --train_precomputed=nq-train.tfrecords-00000-of-00001 \
  --train_num_precomputed=494670 \
  --learning_rate=3e-5 \
  --num_train_epochs=1 \
  --max_seq_length=512 \
  --save_checkpoints_steps=5000 \
  --init_checkpoint=uncased_L-24_H-1024_A-16/bert_model.ckpt \
  --do_train \
  --output_dir=bert_model_output
```
