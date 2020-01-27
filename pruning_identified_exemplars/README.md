# Selective Brain Damage: Measuring the Disparate Impact of Model Pruning

[Website and slides](https://weightpruningdamage.github.io/)

[Full paper](https://arxiv.org/abs/1911.05248)

### tl;dr

Summary: The ability to prune networks with seemingly so little degradation to
generalization performance is puzzling. The cost to top-1 accuracy appears
minimal if it is spread uniformally across all classes, but what if the cost of
model compression is concentrated in only a few classes? Are certain types of
examples or classes disproportionately impacted by pruning?

## What do pruned deep neural networks "forget"?

- We find that certain images, which we term pruning identified exemplars (PIEs), are systematically more impacted by the introduction of sparsity.
- Removing PIE images from the test-set greatly improves top-1 accuracy for both sparse and non-sparse models. These hard-to-generalize-to images tend to be mislabelled, of lower image quality, entail abstract representations, atypical examples or require fine-grained classification.
- Surfacing these examples to domain experts is a valuable interpretability tool and provides actionable information to help improve models through data cleaning or idenifying bias.

## What does this mean for the use of pruned models?

Pruned models are widely used by many real world machine learning applications.
Many of the algorithms on your phone are likely pruned or compressed in some
way. Our results are surprising and suggest that a reliance on top-line metrics
such as top-1 or top-5 test-set accuracy hides critical details in the ways that
pruning impacts model generalization.

However, our methodology also provides new tooling to both identify and correct
for this disparate impact. We are currently extending the research on PIE
exemplars as a tool for cleaning noisy data and an interpretability method to surface
distribution outliers for visual inspection by human.

### Code Respository

This code repository allows for the replication of our findings and also provides a script to compare the robustness of sparse models to common perturbations and natural adversarial images.

## Getting Started: How to evaluate the impact of sparsity on model behavior?

### Download Dataset

To replicate our results, first download [ImageNet](http://www.image-net.org/) and store the dataset as a set tfrecords. This [script](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py) is one possible example of how to do this conversion from raw images to tfrecords.

### Generate a set of checkpoints for pruned and non-pruned models

`save_checkpoint/imagenet_train_eval.py` trains a ResNet-50 on ImageNet with
iterative pruning. This checkpoint is needed before proceeding unto measuring
1) per-class impact, 2) identifying pruning identified exemplars.

### Measuring Per-Class Impact

`per_class_accuracy/aggregate_ckpt_metrics.py` collects per-class evaluation metrics across each saved checkpoint.
`per_class_accuracy/gen_tstatistic.py` takes the aggregated dataframe compiled from `per_class_accuracy/aggregate_ckpt_metrics.py` and computes a per-class t-statistic.

### Identifying PIE: Pruning Identified Exemplars

`pie_dataset_gen/imagenet_predictions.py` generates predictions for every prediction in eval for every stored checkpoint.
`pie_dataset_gen/aggregate_predictions.py` aggregates the predictions saved in `pie_dataset_gen/imagenet_predictions.py` into a single dataframe for every level of sparsity considered.
`pie_dataset_gen/generate_pie_dataset.py`saves a new tfrecords dataframe for pies based upon the output csv from `pie_dataset_gen/aggregate_predictions.py`.

### Measures of Robustness

`robustness_tests/imagenet_corruptions.py` collects per-class evaluation metrics across each saved checkpoint given open source datasets that measure robustness ([ImageNet-C](https://github.com/hendrycks/robustness), [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)).

## Citation

For use of this code or reference of this work, please cite:

`
@ARTICLE{2019Hooker,
       author = {{Hooker}, Sara and {Courville}, Aaron and {Dauphin}, Yann and
         {Frome}, Andrea},
        title = "{Selective Brain Damage: Measuring the Disparate Impact of Model Pruning}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Computer Science - Artificial Intelligence, Computer Science - Computer Vision and Pattern Recognition, Statistics - Machine Learning},
         year = "2019",
        month = "Nov",
       eprint = {1911.05248},
 primaryClass = {cs.LG},
}
`

For any questions about this code please file a github [issue](https://github.com/google-research/google-research/issues) and tag github handle sarahooker.




