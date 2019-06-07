# EDF

Library to convert from EDF file format to tensorflow.Example for model
training. An example application is provided for the Temple University Hospital
(TUH) seizure dataset and has been tested on the same.

## Requirements:
- Bazel 0.19.2.
This was the version that worked with tensorflow 1.13.1, the
most recent one as of June 2019 according to www.tensorflow.org/install/source
- Lullaby
This library is used to parse command-line arguments but it does not have a 
release, hence could not be included in the WORKSPACE.

## Instructions for building with bazel:
- git clone https://github.com/google/lullaby.git lullaby into the parent folder
of eeg_modelling (where the file WORKSPACE exists). This is a workaround until
lullaby has an official release.
- bazel build edf:temple_edf_to_tf_example


