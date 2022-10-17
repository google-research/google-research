# FRMT: Few-shot Region-aware Machine Translation

FRMT (pronounced "format") is a benchmarking dataset for research on fluent
translation into regional language varieties.

The dataset provides human translations of a few thousand English Wikipedia
sentences into regional variants of Portuguese (Brazil and Portugal) and
Mandarin (Mainland and Taiwan).

## Directory structure

-   `lexical_bucket/` is a subset where the English Wikipedia pages were
    selected according to a manual list of terms/phrases that tend to be
    translated differently for the targeted regions. It is useful for assessing
    how well a model handles differences in lexical choice arising from regional
    language differences.
-   `entity_bucket/` is a subset where the English Wikipedia pages are about
    entities that were deemed to have a strong connection to a particular
    targeted region. These serve as adversarial test cases -- a model that
    accurately captures linguistic dialect differences should not, for example,
    produce Portugal Portuguese merely because Lisbon is mentioned.
-   `random_bucket` is a subset from random English Wikipedia pages appearing in
    the "good articles" or "featured articles" collections of Wikipedia.
-   File names and formats:
    -   `{language}_{bucket}_{split}_en_{language}-{REGION}.tsv`:
        -   English sentence \<TAB\> Translation into target language variety

All the English text was sampled from the training split of
[wiki40b/en v1.3.0](https://www.tensorflow.org/datasets/catalog/wiki40b).

## Dataset Statistics

The dataset is split at the document-level into 'exemplar', 'dev', and 'test'
splits. The intended use of the exemplars is as few-shot examples, provided to
the model at inference time to convey which target region is desired.

Number of sentences in the dataset (paired with English):

| Bucket       | Split    | Portuguese | Mandarin
| ------------ | -------- | ---------- | --------
| **Lexical**  | Exemplar | 118        | 173
|              | Dev      | 848        | 524
|              | Test     | 874        | 538
|              |          |            |
| **Entities** | Exemplar | 112        | 104
|              | Dev      | 935        | 883
|              | Test     | 985        | 932
|              |          |            |
| **Random**   | Exemplar | 111        | 111
|              | Dev      | 744        | 744
|              | Test     | 757        | 757
|              |          |            |
| **Total**    | Exemplar | 341        | 388
|              | Dev      | 2527       | 2151
|              | Test     | 2616       | 2227

## Evaluation Code

### Setup

The FRMT evaluation code released here requires Python 3 and the packages
specified in `requirements.txt`. We recommend using an environment manager
like `virtualenv`, e.g.:

```
# From the parent directory of frmt/:
virtualenv -p python3 frmt_env
source frmt_env/bin/activate
pip install -r frmt/requirements.txt
```

Optionally execute `bash frmt/run.sh` to run the tests, which should print
"Success!" if everything is in good order. (`run.sh` depends on `virtualenv`.)

### BLEU, BLEURT, chrF

_Check back soon._ We are in the process of releasing the official evaluation
script for measuring translation quality with BLEU, BLEURT and chrF.

### Lexical Accuracy

`lexical_accuracy_eval.py` computes the lexical accuracy metric, as defined in
the paper.

Below are the commands to reproduce the lexical accuracy numbers reported in the
paper, for the 'test' split of the lexical bucket.

```
BUCKET_DIR=frmt/dataset/lexical_bucket

# Chinese.
# Grab the second column of the TSV file, which contains the gold translations.
cut -f2 ${BUCKET_DIR}/zh_lexical_test_en_zh-CN.tsv > /tmp/zh-cn.txt
cut -f2 ${BUCKET_DIR}/zh_lexical_test_en_zh-TW.tsv > /tmp/zh-tw.txt
python -m frmt.lexical_accuracy_eval --corpus_cn=/tmp/zh-cn.txt --corpus_tw=/tmp/zh-tw.txt
# Expected output: 0.9444

# Portuguese
cut -f2 ${BUCKET_DIR}/pt_lexical_test_en_pt-BR.tsv > /tmp/pt-br.txt
cut -f2 ${BUCKET_DIR}/pt_lexical_test_en_pt-PT.tsv > /tmp/pt-pt.txt
python -m frmt.lexical_accuracy_eval --corpus_br=/tmp/pt-br.txt --corpus_pt=/tmp/pt-pt.txt
# Expected output: 0.9858
```

For more options, see `python -m frmt.lexical_accuracy_eval --help`.


## Citation

If you use or discuss FRMT in your work, please cite [our
paper](https://arxiv.org/abs/2210.00193):

```
@misc{riley2022frmt,
  doi = {10.48550/ARXIV.2210.00193},
  url = {https://arxiv.org/abs/2210.00193},
  author = {Riley, Parker and Dozat, Timothy and Botha, Jan A. and Garcia, Xavier and Garrette, Dan and Riesa, Jason and Firat, Orhan and Constant, Noah},
  title = {{FRMT}: A Benchmark for Few-Shot Region-Aware Machine Translation},
  publisher = {arXiv},
  year = {2022},
}
```

## Support

If you have questions or encounter issues with this dataset, please contact the
authors by email. You're also welcome to open a Github issue, but we might miss
it, because this repository is shared by many Google Research projects.

## License

The dataset is licensed under [CC BY-SA
3.0](http://creativecommons.org/licenses/by-sa/3.0/).
