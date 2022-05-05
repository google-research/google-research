# Code accompanying the Collocated Irradiance Network paper

This package contains code accompanying the paper titled 
"Estimates Of Broadband Upwelling Irradiance From GOES-16 ABI"

## CERES point spread function implementation

Point spread function weights can be calculated for arbitrary locations
quickly in this vectorized numpy implementation. See the unittest for usage
examples.

## Reading training/validation tfrecords

The training/validation data can be experimented with using other model 
architectures. An example of reading the tfrecord format is provided in 
the colab notebook.

## Loading irradiances

Model outputs can be used for researching climate phenomena which benefit from
the spatio-temporal resolution that GOES-16 ABI offers
(2km nominal, 10-15 minute refresh rate).
An example of reading the model outputs (irradiances) is provided
in the colab notebook.
