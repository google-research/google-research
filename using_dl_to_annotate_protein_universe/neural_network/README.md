This code is intended to support the publication "Using Deep Learning to
Annotate the Protein Universe". [preprint link](https://doi.org/10.1101/626507)

A colab showing computing single-neural-network accuracy
on the random split of Pfam seed is in [Neural_network_accuracy_on_random_seed_split.ipynb](https://colab.research.google.com/github/google-research/google-research/blob/master/using_dl_to_annotate_protein_universe/neural_network/Neural_network_accuracy_on_random_seed_split.ipynb)


# Short description of files

-   Neural_network_accuracy_on_random_seed_split.ipynb: as described above.
-   train.py: constructs model, attaches to dataset, runs training.
-   protein_model.py: CNN architecture.
-   hparams_sets.py: hyperparameters.
-   util.py: utilities for, e.g. converting sequences to one-hots.
-   data.py: example consumption of data and conversion into tensors.
