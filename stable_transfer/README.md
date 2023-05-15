## How stable are Transferability Metrics evaluations?

The code in this directory belongs to our paper:

```
@inproceedings={agostinelli22eccv,
  title={How stable are Transferability Metrics evaluations?}
  author={Andrea Agostinelli and Michal Pándy and Jasper Uijlings and Thomas Mensink and Vittorio Ferrari},
  year={2022}
  booktitle={ECCV}
  arxiv={https://arxiv.org/abs/2204.01403}
```

**Abstract**
Transferability metrics is a maturing field with increasing interest, which aims at providing heuristics for selecting the most suitable source models to transfer to a given target dataset, without fine-tuning them all. However, existing works rely on custom experimental setups which differ across papers, leading to inconsistent conclusions about which transferability metrics work best. In this paper we conduct a large-scale study by systematically constructing a broad range of 715k experimental setup variations. We discover that even small variations to an experimental setup lead to different conclusions about the superiority of a transferability metric over another. Then we propose better evaluations by aggregating across many experiments, enabling to reach more stable conclusions. As a result, we reveal the superiority of LogME at selecting good source datasets to transfer from in a semantic segmentation scenario, NLEEP at selecting good source architectures in an image classification scenario, and GBC at determining which target task benefits most from a given source model. Yet, no single transferability metric works best in all scenarios.


### This codebase contains:

* Implementation of 5 transferability metrics (GBC, LEEP, NLEEP, LogME, H-score).
* Implementation of 4 evaluation measures for transferability metrics, as described in the section 3.3 of the paper.
* Pipeline for training multiple model architectures on classification datasets.
* Pipeline for computing the transferability metrics for classification and semantic segmentation tasks.
* Code for performing the qualitative and quantitative evaluations proposed in the paper.
* All experimental results from the paper for reproducing the results, contained in `stable_transfer/transferability/results_analysis/source_datasets_table.csv` and `stable_transfer/transferability/results_analysis/source_model_architectures_table.csv`.

### Training model architectures on classification datasets

All datasets and pre-trained model architectures are loaded using `tensorflow.keras.applications` and `tensorflow_datasets`, hence there is no need to pre-download them. The specific experiment to run can be set in `stable_transfer/transferability/config_training_source_selection.py`. The experiment can be launched by passing the config as flag in `stable_transfer/transferability/main.py`.

### Computing transferability metrics with multiple model architectures on classification datasets

The experiment can be set using `stable_transfer/transferability/config_transferability_source_selection.py`, and launched by passing the config as flag in `stable_transfer/transferability/main.py`.

### Computing transferability metrics with HRNet based models on semantic segmentation datasets

The pre-trained HRNet based models are loaded automatically from `tensorflow_hub`. However, the semantic segmentation datasets need to be downloaded locally beforehand. We refer to the github repository [`google-research/factors_of_influence`](https://github.com/google-research/google-research/tree/master/factors_of_influence) for all the necessary steps. The experiment can then be set using `stable_transfer/transferability/config_transferability_source_selection_segmentation.py`, and launched by passing the config as flag in `stable_transfer/transferability/main.py`.

### Performing qualitative and quantitative evaluations

After calculating all transferability metric and actual performance scores, qualitative and quantitative evaluations are computed in `stable_transfer/transferability/results_analysis/results_analysis.ipynb`. We refer to the provided precomputed data from the paper experiments (`stable_transfer/transferability/results_analysis/source_datasets_table.csv` and `stable_transfer/transferability/results_analysis/source_model_architectures_table.csv`) for the format required by the analysis functions.

