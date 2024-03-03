# Paper

This repo is support for the following publication:

**Title**: Recurrent Convolutional Deep Neural Networks for Modeling Time-resolved Wildfire Spread Behavior<br>
**Authors**: John Burge, Matthew Bonanni, Lily Hu, Matthias Ihme<br>
**Journal**: Fire Technology, 2023.<br>

The predictions made in this work take the state of a fire at timestep *t*, and
predict what the state of the fire at timestep *t+1* is.  This repo only contains
the code for a Google colab.  To use this colab, you will need to first copy the model checkpoint files and the dataset files from Google Cloud to a local machine, and then into the colab.  This document provides detailed instructions on how to accomplish that.

# Google cloud

All of the data and the model checkpoints are located on Google cloud.  To access the data, you can install the google cloud CLI, which this document will assume you've done (access to the cloud data can also be performed on the Google Cloud UI for users familiar with it).  For instructions on installing Google Cloud CLI:

https://cloud.google.com/sdk/docs/install
# Data:

You can view all the datasets via:

```
gsutil ls gs://wildfire_conv_lstm/data
```

There is a single subdirectory for each dataset:

* single_fuel
* multiple_fuel
* california
* california_wn

The data folders each contain 10 sub directories, each containing roughly 10% of the data for that dataset.  E.g.:

```
gsutil ls gs://wildfire_conv_lstm/data/california_wn
```

Results in:

```
gs://wildfire_conv_lstm/data/california_wn/00_$folder$
gs://wildfire_conv_lstm/data/california_wn/01_$folder$
gs://wildfire_conv_lstm/data/california_wn/02_$folder$
gs://wildfire_conv_lstm/data/california_wn/03_$folder$
gs://wildfire_conv_lstm/data/california_wn/04_$folder$
gs://wildfire_conv_lstm/data/california_wn/05_$folder$
gs://wildfire_conv_lstm/data/california_wn/06_$folder$
gs://wildfire_conv_lstm/data/california_wn/07_$folder$
gs://wildfire_conv_lstm/data/california_wn/08_$folder$
gs://wildfire_conv_lstm/data/california_wn/09_$folder$
gs://wildfire_conv_lstm/data/california_wn/00/
gs://wildfire_conv_lstm/data/california_wn/01/
gs://wildfire_conv_lstm/data/california_wn/02/
gs://wildfire_conv_lstm/data/california_wn/03/
gs://wildfire_conv_lstm/data/california_wn/04/
gs://wildfire_conv_lstm/data/california_wn/05/
gs://wildfire_conv_lstm/data/california_wn/06/
gs://wildfire_conv_lstm/data/california_wn/07/
gs://wildfire_conv_lstm/data/california_wn/08/
gs://wildfire_conv_lstm/data/california_wn/09/
```

The entries with "_$folder$" in them are artifacts of Google Cloud and can be ignored.

Each of these directories contains a single sharded TFRecordio, which is a data format used by Tensorflow to store training data.  Each sharded TFRecordio is a collection of ~200 files.  Each of these files contains zero or more data points, and each file contains a random set of the data points for that dataset.

The files in each directory will have the following tags in their filename:

* 'test' - This file was used as a testing dataset in the paper (for subdirs 00, 01, 02, 03, 04) or a validation dataset (for subdirs 05, 06, 07, 08, 09).
* 'train' - This file was used to train a model.
* 'temporal' - This file contains temporal data points for the EPD-ConvLSTM model.
* 'static' - This file contains static data for the EPD model.

Other tokens in the filenames can be ignored.

All of the files are publicly accessible.  In order to use them in the colab, first you'll need to copy the files to a local directory.  In our example, we'll copy over a single file from the California Wind Ninja dataset:

```
# Make local directory.
mkdir -p ~/tmp/test

# Copy over one shard of the sharded TFRecordIO.
gsutil cp gs://wildfire_conv_lstm/data/california_wn/00/real_wn_test_temporal.tfr-00164-of-00165 ~/tmp/test
```

# Model Checkpoints

You can view all the model check points:

```
gsutil ls gs://wildfire_conv_lstm/checkpoints
```

Like the data files, there are 4 groups of models for each of the 4 datasets evaluated in the paper.  Each subdirectory contains two models:

* epd: The EPD model used to make predictions on static data points.
* lstm: The EPD-ConvLSTM model used to make predictions on temporal data points.

Since the data file we copied contains the tag 'temporal', we need the EPD-ConvLSTM model checkpoint stored in lstm.h5:

```
gsutil cp gs://wildfire_conv_lstm/checkpoints/california_wn/lstm.h5 ~/tmp/test
```

# Colab

The code for performing an inference with the dataset on the model is given in a colab in the following github repo:

https://github.com/google-research/google-research/tree/master/wildfire_conv_lstm

To use this code, first head to the Google colab main page:

https://colab.research.google.com/

On the opening screen, select the "GitHub" tab, and then search for 'google-research'.  Then, in the "Repository" dropdown, select the "google-research/google-research" repository (the repetition is not a typo). Then, in the path textbox, select the "wildfire_conv_lstm/wildfire_conv_lstm.ipynb" entry.

That should bring up the code in a colab.  Don't forget to connect to a server after loading up the colab.

Next, copy over the data file and the model file to the colab server, using the colab file UI.  In this example, the files were copied over to the /tmp directory in the root directory of the colab server (any directory will do).  Execute the first 3 cells to get all the code loaded.  Then in the final cell, enter the following fields:

input_file: /tmp/real_wn_test_temporal.tfr-00164-of-00165<br>
input_model: /tmp/lstm.h5<br>

Note that the model is an 'lstm' model and the file is a 'temporal' file.  If the model was a 'epd' model, then the corresponding file needed would be the 'static' file.

Finally, run the last cell, which will then result in a prediction being made.  Three plots are provided showing: 1) the label associated with this data point.  2) the prediction made by the model.  3) The error between the label and the prediction.

## Normalization and notes on Data

Like most ML models, the data fed into the model is normalized before training.  All of the data files provided have had their features normalized.  The first three fields in the data are measurements of vegetation, and their values naturally already ranged between 0.0 and 1.0.  All other channels were normalized to unit variance and zero mean.  In each subdirectory containing a set of data (the 00, 01, 02, etc.), there is a normalization file that provides a tuple containing coefficients that can be used to unnormalize the results.  The three-tuple includes:

1) An int providing the count of rows used during norm estimation.
2) The observed std
3) The observed mean

Normalization files can be found in each of the subdirectories where the sharded TFRecordIO files are stored.

The data was originally generated by running FARSITE, but additional code was written to reformat the output into data that is more conducive to training an ML model with.  The following github repository was used to wrap FARSITE in order to perform many simulations:

https://github.com/IhmeGroup/farsite_utils

The code in that github does not fully generate the files we are sharing.  Instead, that script generated an intermediate file format, which was then converted into the files that we are sharing.  Unfortunately, technical limitations prevented the sharing of that piece of code.  But that code primarily just normalized the data (zero mean unit variance) and did not introduce any new data--it merely massaged the data a bit.  So while this github code will not fully regenerate the data in this paper, the underlying tools in the repo can be very useful for helping wrap the FARSITE code for making large numbers of FARSITE simulations.

# Code Snippets

What the colab does not facilitate is being able to perform additional runs of FARSITE to generate additional cases to apply the model to.  While it was not feasible for us to release our code for processing the output of FARSITE, we have released a high-level document that summarizes that work.  See the file supplemental_pseudocode.txt for details.  We have also released snippets of the python code that generates the tf.train.Example(s) that the colab uses.  Unfortunately, these snippets cannot directly be run, but they should provide all of the details needed to duplicate our work.  See create_data_snippets.py.snippets for the python code snippets.

This code starts from the numpy files that are created by the code we used to wrap farsite (described above).

The colab referred above could be used to analyze the structure of the models.  However, we have also provided some code snippets that show how the layers of those models were constructed.  See create_epd_snippets.py.snippets and create_epd_conv_lstm_snippets.py.snippets.  As before, this code is not fully functional, but should provide most of the details required for a user to construct similar models for training.  Likely, some knowledge of Tensorflow and Keras will be required to understand the code snippets and use them as a potential source of code.
