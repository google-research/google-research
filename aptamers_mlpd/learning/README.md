# TensorFlow models for Aptamers

## Running these models

See the playbook (playbook.md) for instructions.

## Binned and SuperBin Labels

As described in the manuscript the binned model used a ternary label to
represent each sequence for each subexperiment based on the positive bead
fraction. The ternary labels with fraction cutoffs were:

-   0 = [0, .1)
-   1 = [.1, .9]
-   2 = (.9, 1.0]

For the three subexperiments in PD round 2 these were labeled as 'low_3bins',
'med_3bins', 'high_3bins'.

These bin labels were then passed into the super_bin which created a label of
either:

-   -1: ambiguous
-   0: all 0
-   1: all 0 except low_3bins = 1
-   2: all 0 except low_3bins = 2 or low_3bins = 1, med_3bins = 1
-   3: low_3bins= 2, med_3bins=1
-   4: low_3bins=2, med_3bins=2 or low_3bins=2, med_3bins=1, high_3bins=1
-   5: low_3bins=2, med_3bins=2, high_3bins=1
-   6: low_3bins=2, med_3bins=2, high_3bins=2

For clarity, an ipynb 'create_binned_and_super_bin_labels.ipynb' has been
included in this folder which can go from the provided
pd_clustered_input_data_manuscript.csv to both the binned and super_bin labels.

## Labeled axes

We use LabeledTensor[https://github.com/tensorflow/tensorflow/tree/v1.15.0/tensorflow/contrib/labeled_tensor] to keep track of meaningful
axis name and labels on our tensors. The `LabeledTensor` objects defined in this
module share a common set of `labeled_tensor.Axis` objects:

- `batch_axis`: an axis with name `'batch'` and no axis labels denoting batches
  of training/evaluation examples.
- `input_position_axis`: an axis with name `'input_position'` and labels
  giving the integer number of nucleic acid offsets from the 5-prime end
  (namely, `[0, 1, 2, ..., sequencing_length - 1]`).
- `input_channel_axis`: an axis with the name `'input_channel'` and labels for
  the channels corresponding to each input feature (e.g., `['A', 'T', 'G', 'C']`
  if not using secondary structure).
- `output_axis`: an axis with the name `'output'` and labels correpsonding to
  the name of each physical measurement/assay that could be used for either
  training or validation. By construction, labels match `SequencingReads.name`
  fields from the
  `selection_pb2.Experiment` proto in aptamers_paper1/selection.proto.
- `logit_axis`: an axis used to label each distinct prediction from the
  feedforward network part of the model. The name of this axis and its labels
  depend on the particular output layer model.
- `target_axis`: an axis with the name `'target'` used to label loss targets for
  training. Labels for loss targets should be a subset of the measurements
  provided by `output_axis`.

`batch_axis`, `input_position_axis`, `input_channel_axis` and `output_axis` are
produced by preprocessing routines in aptamers_paper1/learning/data.py.
`logit_axis` and `target_axis` are created by
 output layer models in aptamers_paper1/learning/output_layers.py.
