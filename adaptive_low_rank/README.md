# AdaRank: Disagreement Based Module Rank Prediction for Low-rank Adaptation

With the rise of language and multimodal models of ever-increasing size, pretraining a general-purpose foundational model and adapting it to downstream tasks has become common practice. To this end, adaptation efficiency can be a critical bottleneck given the large model sizes, hence efficient finetuning methods such as LoRA have become prevalent. However, LoRA is typically applied with the same rank across all model layers, despite mounting evidence from transfer learning literature that during finetuning, later layers diverge more from pretrained weights. Inspired by the theory and observations around feature learning and module criticality, we develop a simple model disagreement based technique to predict the rank of a given module relative to the other modules. Empirically, AdaRank generalizes notably better on unseen data than uniform ranks. Compared to prior work, AdaRank has the unique advantage of leaving the pretraining and adaptation stages completely intact: no need for any additional objectives or regularizers, which can hinder adaptation accuracy and performance.

Getting started with AdaRank is simple. For example:

```
python3 train.py --dataset_name=trec --target_key=label-coarse --num_classes=6 --seed=538 --decay_steps=200 --num_steps=500 --ranks="1,3,7,9,9,8,8,8,8,10,10,8" --replaced_module=value
```

Command line options can be specified as:
```
  --batch_size: Batch size
    (default: '256')
    (an integer)
  --dataset_name: Dataset name for train and test.
    (default: 'yelp_polarity_reviews')
  --decay_steps: Number of decay steps for the optimizer
    (default: '1000')
    (an integer)
  --learning_rate: Learning rate.
    (default: '0.001')
    (a number)
  --model_name: Name of the transformer model to use. E.g. bert, roberta
    (default: 'bert')
  --num_classes: Number of classes
    (default: '2')
    (an integer)
  --num_steps: Number of training steps
    (default: '500')
    (an integer)
  --ranks: List of ranks of the low-rank layers. If specifying
    ranks of the low-rank layers for all modules. Must be in order "query",
    "key", "value", "dense". E.g.: 1,3,3,3,4,5,5,6,7,7,7,7
    5,6,5,5,6,6,5,6,6,6,6,6 2,6,11,14,15,14,13,14,13,16,16,13
    7,8,8,8,7,7,6,8,8,7,8,8. Separate each module by a space.
    (default: '4,4,4,4,4,4,4,4,4,4,4,4')
    (a whitespace separated list)
  --replaced_module: Name of the module to be low-rank adapted, can be key,
    query, value, dense, or all.
    (default: 'query')
  --seed: Random seed
    (default: '21')
    (an integer)
  --target_key: Name of the prediction target.
    (default: 'label')
  --task_type: Task type can be classification or regression.
    (default: 'classification')
  --text_key: Name of the text feature.
    (default: 'text')
  --work_dir: Working directory for logging and saving artifacts.
```

Disclaimer: This is not an officially supported Google product.

