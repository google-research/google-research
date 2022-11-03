## SMART: Sentences as Basic Units for Summary Evaluation

This directory contains tools for using SMART to evaluate texts produced
by systems, given the source document and the reference summaries.

Link to paper: https://arxiv.org/pdf/2208.01030.pdf

### Run SMART Evaluation

SMART can be run programmatically. For example:

```
matcher = matching_functions.chrf_matcher
smart_scorer = scorer.SmartScorer(matching_fn=matcher)
score = smart_scorer.smart_score(reference, candidate)
```

Here, `score` is a dictionary containing SMART (1/2/L) scores.

### Replicate SummEval results in the paper

You first need to download the necessary datasets:
1. [BARTScore data](https://github.com/neulab/BARTScore/tree/main/SUM/SummEval) (you need to unpickle and save it again as a json file)
2. [SummEval data](https://drive.google.com/file/d/1d2Iaz3jNraURP1i7CfTqPIj8REZMJ3tS/view)

You also need to download the precomputed scores for model-based matching functions (e.g., BLEURT, BERTScore, and T5-ANLI). In the terminal, follow the instructions and install [gsutil](https://cloud.google.com/storage/docs/gsutil_install). Then run:

```
gsutil cp -r gs://gresearch/SMART ./
```

Then, finally, run the following:

```
python summeval_experiments.py --bartscore_file=${BARTSCORE_PATH} --summeval_file=${SUMMEVAL_PATH} -- output_file=${OUTPUT_PATH}
```
