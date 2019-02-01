# Evaluation of Explainability Methods in Deep Neural Networks

*ROAR is a benchmark to evaluate the approximate accuracy of interpretability
methods that estimate input feature importance in deep neural networks.*

This directory contains the code needed to implement **ROAR**, RemOve And
Retrain, a benchmark to evaluate the approximate accuracy of feature importance
estimators.

## ROAR: Remove and Retrain

### tl;dr

- Many interpretability methods for deep neural networks estimate the feature importance of pixels in an image to the model prediction.
- For sensitive domains like healthcare, autonomous vehicles and credit scoring these estimates must be both 1) meaninful to a human and 2), highly accurate, as an incorrect explanation of model behavior may have intolerable costs on human welfare.
-   ROAR is concerned with 2). ROAR is a benchmark to measure the approximate accuracy of feature importance estimators.
    ROAR removes the fraction of input features deemed to be most important
    according to each estimator and measurea the change to the model accuracy
    upon retraining. The
    most accurate estimator will identify inputs as important whose removal
    causes the most damage to model performance relative to all other
    estimators.
    
For more information about ROAR, you can find our paper [here](https://arxiv.org/abs/1806.10758).
In addition, slides about ROAR [here](https://drive.google.com/file/d/1zEgjJBkcsPM3J6IkAYGfoEZTsfmk5Csz/view).

## Getting Started: How to benchmark an interpretability method using ROAR?

### Download Dataset

In our paper, we evaluate the accuracy of model explanations for image classification on birdsnap, ImageNet and Food101 datasets.

These are publicaly available datasets. To replicate our results, first download the dataset you plan to evaluate performance on and store the dataset as a set tfrecords.

### Generate Feature Importance Estimates for each dataset


```saliency_data_gen/dataset_generator.py``` generates feature importance estimates for every image in the tfrecords dataset. Feature importance estimates covered: integrated gradients (IG), sensitivity heatmaps (SH), guided backprop (GB), and the smoothgrad, squared, smoothgrad^2 and vargrad versions of IG, SH, GB.

The script produces a new set of tfrecords (the data input for ```interpretability_benchmark/train_resnet.py```) with both the original image and the feature importance estimate.

### Train model on modified dataset

```interpretability_benchmark/train_resnet.py``` trains a resnet_50 on the modified tfrecords dataset (images where a fraction of pixels have been removed).

## Citation
If you are using ROAR code you may cite:
```
@article{2018explainabilityeval,
  author    = {Sara Hooker and
               Dumitru Erhan and
               Pieter{-}Jan Kindermans and
               Been Kim},
  title     = {Evaluating Feature Importance Estimates},
  year      = {2018},
  url       = {http://arxiv.org/abs/1806.10758},
}
```




