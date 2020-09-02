# DrFact

## Installation

```bash
# Note that we use TF 1.15. This is because we use the tf.contrib package,
# which was removed from TF 2.0.
conda create --name drfact python=3.7
conda activate drfact
pip install tensorflow==1.15
# pip install tensorflow-gpu==1.15.0
# We use TensorFlow Hub to download BERT, which is used to initialize DrFact.
pip install tensorflow-hub bert-tensorflow
pip install spacy -U
python -m spacy download en_core_web_lg
```

## Corpus Preprocessing

### Download the GenericsKB corpus.

Link: https://allenai.org/data/genericskb

Put the `GenericsKB-Best.tsv` file under the `./knowledge_corpus/` folder.
Note that the `language` and `knowledge_corpus` folder are at the same level.

Run `CorpusPrepro.ipynb`, `NounChunkVocab.ipynb` and `ChangeFormat.ipynb`.
All the preprocessed data here should be at `/path/to/knowledge_corpus/`.


## Indexing

Put the BERT ckpts `bert/pretrained_models/wwm_uncased_L-24_H-1024_A-16/*` to `/path/to/bert_large/`.

Set up the environment.
```
CORPUS_PATH=/path/to/knowledge_corpus/
BERT_PATH=/path/to/bert_large/
INDEX_PATH=/path/to/local_drfact_index
```

### Basic data preprocessing
```
python index_corpus.py \
--do_preprocess \
--entity_file ${CORPUS_PATH}/gkb_best.vocab.txt \
--wiki_file ${CORPUS_PATH}/gkb_best.drkit_format.jsonl \
--max_entity_length 5 \
--max_mentions_per_doc 20 \
--tokenizer_type bert_tokenization \
--vocab_file ${BERT_PATH}/vocab.txt \
--multihop_output_dir ${INDEX_PATH}/drfact_output_bert200 \
--alsologtostderr
```

### Fact2Fact matrix computation
Run indexing in parallel (`num_shards=3`).
```
for (( c=0; c<=3; c++ ))
do
   python fact2fact_index.py \
    --do_preprocess \
    --entity_file ${CORPUS_PATH}/gkb_best.vocab.txt \
    --wiki_file ${CORPUS_PATH}/gkb_best.drkit_format.jsonl \
    --fact2fact_index_dir ${INDEX_PATH}/fact2fact_index \
    --multihop_output_dir ${INDEX_PATH}/drfact_output_bert200 \
    --num_shards 3 --my_shard $c \
    --alsologtostderr
done
```

Combine the single files.
```
python fact2fact_index.py \
  --do_combine \
  --entity_file ${CORPUS_PATH}/gkb_best.vocab.txt \
  --wiki_file ${CORPUS_PATH}/gkb_best.drkit_format.jsonl \
  --fact2fact_index_dir ${INDEX_PATH}/fact2fact_index \
  --multihop_output_dir ${INDEX_PATH}/drfact_output_bert200 \
  --num_shards 3 \
  --alsologtostderr
```

### Embedding the facts with BERT.

(Note that we can also embed mentions by replace `do_embed_facts` with `do_embed_mentions`).

Run computation (`num_shards=1` if only have one gpu).
```
python index_corpus.py \
--do_embed \
--do_embed_facts \
--entity_file ${CORPUS_PATH}/gkb_best.vocab.txt \
--wiki_file ${CORPUS_PATH}/gkb_best.drkit_format.jsonl \
--max_entity_length 5 \
--max_mentions_per_doc 20 \
--tokenizer_type bert_tokenization \
--bert_ckpt_dir ${BERT_PATH}/ \
--vocab_file ${BERT_PATH}/vocab.txt \
--bert_config_file ${BERT_PATH}/bert_config.json \
--ckpt_name bert_model.ckpt --embed_prefix bert_large \
--multihop_output_dir ${INDEX_PATH}/drfact_output_bert200 \
--predict_batch_size 128 \
--max_seq_length 128 \
--doc_layers_to_use -1 \
--doc_aggregation_fn concat \
--qry_layers_to_use 4 \
--qry_aggregation_fn concat \
--max_query_length  64 \
--projection_dim 200 \
--doc_stride 128 \
--num_shards 1 \
--alsologtostderr
```

Run combination.
```
python index_corpus.py \
--do_combine \
--do_embed_facts \
--max_entity_length 5 \
--max_mentions_per_doc 20 \
--multihop_output_dir ${INDEX_PATH}/drfact_output_bert200 \
--num_shards 1 \
--projection_dim 200 \
--tokenizer_type dummy \
--shards_to_combine 1 \
--alsologtostderr
```

Clean the data:

```
PREFIX=bert_large
rm ${INDEX_PATH}/drfact_output_bert200/${PREFIX}_fact_feats_*.*
```

## Dataset Processing

Run the `QADataPrepro.ipynb` for preprocessing the datasets.

Then, we link the concepts in the questions and answers:
```
ORI_DATA_DIR=/path/to/datasets/ARC-Easy/;
VOCAB=${CORPUS_PATH}/gkb_best.vocab.txt

SPLIT=arc_easy_train NUM_CHOICES=-1;  \
python link_questions.py \
  --csqa_file $ORI_DATA_DIR/${SPLIT}_processed.jsonl \
  --output_file $DATA_DIR/linked_${SPLIT}.all.jsonl \
  --indexed_concept_file  ${VOCAB} \
  --do_filtering ${NUM_CHOICES} \
  --alsologtostderr

SPLIT=arc_easy_dev NUM_CHOICES=3; \
python link_questions.py \
  --csqa_file $ORI_DATA_DIR/${SPLIT}_processed.jsonl \
  --output_file $DATA_DIR/linked_${SPLIT}.all.jsonl \
  --indexed_concept_file ${VOCAB} \
  --do_filtering ${NUM_CHOICES} \
  --alsologtostderr
```


## Data Analysis

## Model Training and Evaluation

```
bash drfact.sh
```

