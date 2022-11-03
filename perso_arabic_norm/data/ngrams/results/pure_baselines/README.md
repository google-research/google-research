## Results without line differences

This directory has the results that do *NOT* take into account line diffs.

### Example:

```bash
python experimental/nlp/sweet/roc/ngram_train_and_eval.py \
  --corpus_file /usr/local/google/sweet/perso_arabic/wiki/text/kswiki-20211101-pages-meta-current_thr0.6.txt.bz2 \
  --num_trials 100 --output_model_dir /usr/local/google/sweet/perso_arabic/wiki/results/baselines
```
