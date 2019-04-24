# Processing of microarray data for transcription networks in yeast

This python notebook is a companion to [Time-resolved genome-scale profiling
reveals a causal expression network][paper].


## Downloading the data

There are four data files that go along with the [paper][paper]:

`yeast_raw_data_table_20180826.tsv`: "Raw" microarray data.

`yeast_data_table_20180826.tsv`: "Processed" microarray data.

`insample_coefs_20170601.csv`: Prediction model before validation experiments.

`insample_coefs_20180826.csv`: Prediction model retrained with validation
experiments.

## Datasets described in the published paper

`raw`: column `r_g_ratio` in `yeast_raw_data_table_20180826.tsv`.

`cleaned`: column `log2_cleaned_ratio` in `yeast_data_table_20180826.tsv`.

`thresholded`: column `log2_cleaned_ratio_zth2dfilt` in `yeast_data_table_20180826.tsv`.


## Running the notebook

Once the data is downloaded locally, edit the notebook to reflect the location:

`datadir = /path/to/datafiles`

1. Run the cells in **Common functions**.

2. If desired, run the cells in **Process the "raw" data**.  This will verify
   the processed data can be reconstructed from the raw data.

3. Run "Read processed data".  Only run **Verify the processing procedure** if
   the **Process the "raw" data** was run.

## Modeling

The **Linear modeling** section shows how to construct design matrices for
linear modeling from the processed data.  A simple example using `sklearn` is
provided.  Note that in the [paper] we used one of the many incompatible
wrappers to the Fortran `glmnet`.

The **Prediction model** section shows how to use the model coefficients to
construct the predicted datasets.  This code also separates the full predictions
into their component parts due to each individual coefficient.

[paper]: https://www.biorxiv.org/
