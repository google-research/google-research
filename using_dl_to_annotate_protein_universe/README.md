This code is intended to support the publication "Using Deep Learning to
Annotate the Protein Universe". [preprint link](https://doi.org/10.1101/626507)

## Short description of files provided

A colab demonstrating usage of the available neural networks is in [Using_Deep_Learning_to_Annotate_the_Protein_Universe.ipynb](https://colab.research.google.com/github/google-research/google-research/blob/master/using_dl_to_annotate_protein_universe/Using_Deep_Learning_to_Annotate_the_Protein_Universe.ipynb).
This shows both prediction of Pfam families as well as how to produce an
embedding for a domain.

Code in the following directories is intended for documentation purposes, and is
not necessarily runnable. More information on these files is in a README.md in
each folder. We used tensorflow-gpu v1.15.4 and python v3.7 in this project.

evaluation/

-   model accuracy
-   statistical significance
-   inference confidence
-   accuracy stratification based on sequence similarity, family size, etc.

hmm_baseline/

-   running hmmer and phmmer

neural_network/

-   constructing CNN
-   training CNN

## Availability of trained models

Trained models are available in
[Google Cloud Storage](https://console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/pfam/models/single_domain_per_sequence_zipped_models)

```
└── single_domain_per_sequence_zipped_models
    ├── full_random_32.0              # Pfam full random test-train split.
    ├── full_random_for_pfam_n_34.0   # Models used for [Pfam-N v34.0](https://xfam.wordpress.com/2021/03/24/google-research-team-bring-deep-learning-to-pfam/)
    ├── seed_clustered_32.0           # Pfam seed clustered test-train split.
    └── seed_random_32.0              # Pfam seed random test-train split.
```

They are in TensorFlow SavedModel format.

## Notes on data availability

Data for the random seed split is available on Kaggle at
https://www.kaggle.com/googleai/pfam-seed-random-split

Data for the random and clustered splits is available in google cloud storage:
https://console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/pfam
