# Selective Brain Damage: Measuring the Disparate Impact of Model Pruning

### tl;dr

- Neural network pruning techniques have demonstrated it is possible to remove the majority of weights in a network with surprisingly little degradation to test set accuracy. However, this measure of performance conceals significant differences in how different classes and images are impacted by pruning.
- We find that certain images, which we term pruning identified exemplars (PIEs), are systematically more impacted by the introduction of sparsity.
- Removing PIE images from the test-set greatly improves top-1 accuracy for both sparse and non-sparse models. These hard-to-generalize-to images tend to be mislabelled, of lower image quality, entail abstract representations, atypical examples or require fine-grained classification.

This code repository allows for the replication of our findings and also provides a script to compare the robustness of sparse models to common perturbations and natural adversarial images.

## Getting Started: How to evaluate the impact of sparsity on model behavior?

### Download Dataset

To replicate our results, first download [ImageNet](http://www.image-net.org/) and store the dataset as a set tfrecords. This [script](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py) is one possible example of how to do this conversion from raw images to tfrecords.

### Generate a set of checkpoints for pruned and non-pruned models

```save_checkpoint/imagenet_train_eval.py``` trains a ResNet-50 on ImageNet with
iterative pruning. This checkpoint is needed before proceeding unto measuring
1) per-class impact, 2) identifying pruning identified exemplars.

### Measuring Per-Class Impact

```per_class_accuracy/aggregate_ckpt_metrics.py``` collects per-class evaluation metrics across each saved checkpoint.
```per_class_accuracy/gen_tstatistic.py``` takes the aggregated dataframe compiled from ```per_class_accuracy/aggregate_ckpt_metrics.py``` and computes a per-class t-statistic.

### Identifying PIE: Pruning Identified Exemplars

```pie_dataset_gen/imagenet_predictions.py``` generates predictions for every prediction in eval for every stored checkpoint.
```pie_dataset_gen/aggregate_predictions.py``` aggregates the predictions saved in ```pie_dataset_gen/imagenet_predictions.py``` into a single . dataframe for every level of sparsity considered.
```pie_dataset_gen/generate_pie_dataset.py```saves a new tfrecords dataframe for pies based upon the output csv from ```pie_dataset_gen/aggregate_predictions.py```.

### Measures of Robustness

```robustness_tests/imagenet_corruptions.py``` collects per-class evaluation metrics across each saved checkpoint given open source datasets that measure robustness ([ImageNet-C](https://github.com/hendrycks/robustness), [ImageNet-A](https://github.com/hendrycks/natural-adv-examples).

## Citation
We will be releasing our pre-print shortly and will update here with the link and citation to our work.
Authors: Sara Hooker, Yann Dauphine, Aaron Courville, Andrea Frome

For any questions about this code please file an github [issue](https://github.com/google-research/google-research/issues) and tag github handles sarahooker. We welcome pull requests which add additional interpretability methods to be benchmarked or improvements to the code.




