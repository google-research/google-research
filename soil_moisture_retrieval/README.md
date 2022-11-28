# A Machine Learning Data Fusion Model for Soil Moisture Retrieval

This repository is the official supporting codebase of [A Machine Learning Data Fusion Model for Soil Moisture Retrieval](https://arxiv.org/abs/2206.09649). We release the following code, data and models along with our paper.

## Overview

- Dataset
  - A partial dataset containing paired input remote sensing and geo-physical sources along with soil moisture readings for a few sensor networks. ISMN based sensor networks do not allow for data redistribution and hence we have not included in-situ readings from ISMN and a few other networks that don't allow re-distribution in the data set (hence the term "partial").
  - A script (combine_ismn_data.py) to combine ISMN soil moisture readings with the dataset we provide. This allows the user to create a "full" dataset by merging data we provide along with ISMN data which can be downloaded from https://ismn.geo.tuwien.ac.at/
  - If you use our dataset, please cite us using

  ```
  @misc{batchu2022-dk,
    doi = {10.48550/ARXIV.2206.09649},
    url = {https://arxiv.org/abs/2206.09649},
    author = {Batchu, Vishal and Nearing, Grey and Gulshan, Varun},
    title = {A Machine Learning Data Fusion Model for Soil Moisture Retrieval},
    publisher = {arXiv},
    year = {2022},
    copyright = {arXiv.org perpetual, non-exclusive license}
  }

  ```
- Model
  - A tensorflow (TF-1) frozen graph based pre-trained model corresponding to the best performing soil moisture model we train in the paper is provided.
  - A script (run_sample_inference.py) to run sample inference on the trained model using either the "partial" dataset or the "full" dataset users can create by using the "combine_ismn_data.py" script.

## Setup

To install requirements in a virtual environment, please perform the following:

```setup
pip install virtualenv
virtualenv <VIRTUAL_ENV_NAME>
cd <VIRTUAL_ENV_NAME>
source bin/activate
cd ..
pip install -r requirements.txt
```

Once the above is done, you should be able to follow along on the sections below and run the scripts provided.

## Soil moisture dataset

With the paper we release a partial dataset hosted on Google Cloud Storage at "eoscience-public/soil_moisture_retrieval_data" under the CC-BY 4.0 [LICENSE](https://storage.googleapis.com/eoscience-public/soil_moisture_retrieval_data/LICENSE.md), containing 500 shards where a shard can be downloaded as follows https://storage.googleapis.com/eoscience-public/soil_moisture_retrieval_data/data-00000-of-00500.tfrecord.gz, consisting of all the inputs but only a subset of soil moisture labels (corresponding to non-ISMN in-situ sensors). The data consists of sharded [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) files (compressed with a gzip format) where each TFRecord consists of multiple TFExamples.

Note: Not all the fields present in the TFExample are used in our models but we provide them to make it easier to explore additional inputs for further research.

The following fields are present in a TFExample datapoint:

- Properties
  - center_lat (Latitude of the in-situ sensor)
  - center_long (Longitude of the in-situ sensor)
  - sensor_id
  - size (Number of pixels in the height/width of the image)
  - timestamp (Timestamp of the in-situ soil moisture data)
  - utm_easting_northing_image_polygon (Bounding polygon in UTM easting/northing)
  - utm_zone (The UTM zone as a string)
  - mask
- Aster
  - aster_B13 (Aster band 13)
  - aster_B14 (Aster band 14)
  - aster_time_millis_diff (Aster time delta from in-situ label)
  - aster_timestamp  (Timestamp of the ASTER image)
- Copernicus Land Cover
  - cgls_lc_crops-coverfraction
  - cgls_lc_discrete_classification
  - cgls_lc_grass-coverfraction
  - cgls_lc_moss-coverfraction
  - cgls_lc_time_millis_diff (CGLS time delta from in-situ label)
  - cgls_lc_timestamp
- GLDAS
  - gldas_evaporation
  - gldas_rainfall
  - gldas_sm
  - gldas_snow_depth
  - gldas_temp
  - gldas_time_millis_diff (GLDAS time delta from in-situ label)
  - gldas_timestamp
- MODIS Leaf Area Index
  - modis_leaf_area_Fpar
  - modis_leaf_area_Lai
  - modis_leaf_area_time_millis_diff (MODIS LAI time delta from in-situ label)
  - modis_leaf_area_timestamp
- NASA DEM
  - nasa_dem_elevation
  - nasa_dem_time_millis_diff (NASA DEM time delta from in-situ label)
  - nasa_dem_timestamp
- Sentinel-1
  - sentinel_1_VH
  - sentinel_1_VV
  - sentinel_1_angle
  - sentinel_1_max_VH (Max of VH over the last 1 year at the given location)
  - sentinel_1_max_VV
  - sentinel_1_max_angle
  - sentinel_1_mean_VH
  - sentinel_1_mean_VV
  - sentinel_1_mean_angle
  - sentinel_1_min_VH
  - sentinel_1_min_VV
  - sentinel_1_min_angle
  - sentinel_1_orbit_pass
  - sentinel_1_time_millis_diff (Sentinel-1 time delta from in-situ label)
  - sentinel_1_timestamp
- Sentinel-2
  - sentinel_2_B11
  - sentinel_2_B12
  - sentinel_2_B2
  - sentinel_2_B3
  - sentinel_2_B4
  - sentinel_2_B5
  - sentinel_2_B6
  - sentinel_2_B7
  - sentinel_2_B8
  - sentinel_2_cloud_percentage
  - sentinel_2_max_B11 (Max of B11 over the last 1 year at the given location)
  - sentinel_2_max_B12
  - sentinel_2_max_B2
  - sentinel_2_max_B3
  - sentinel_2_max_B4
  - sentinel_2_max_B5
  - sentinel_2_max_B6
  - sentinel_2_max_B7
  - sentinel_2_max_B8
  - sentinel_2_max_sen_2_mask
  - sentinel_2_mean_B11 (Mean of B11 over the last 1 year at the given location)
  - sentinel_2_mean_B12
  - sentinel_2_mean_B2
  - sentinel_2_mean_B3
  - sentinel_2_mean_B4
  - sentinel_2_mean_B5
  - sentinel_2_mean_B6
  - sentinel_2_mean_B7
  - sentinel_2_mean_B8
  - sentinel_2_mean_sen_2_mask
  - sentinel_2_min_B11 (Min of B11 over the last 1 year at the given location)
  - sentinel_2_min_B12
  - sentinel_2_min_B2
  - sentinel_2_min_B3
  - sentinel_2_min_B4
  - sentinel_2_min_B5
  - sentinel_2_min_B6
  - sentinel_2_min_B7
  - sentinel_2_min_B8
  - sentinel_2_min_sen_2_mask
  - sentinel_2_sen_2_mask
  - sentinel_2_time_millis_diff (Sentinel-2 time delta from in-situ label)
  - sentinel_2_timestamp
- SMAP
  - smap_sm (SMAP soil moisture estimate at the in-situ sensor location)
  - smap_time_millis_diff (SMAP time delta from in-situ label)
  - smap_timestamp (Timestamp of the SMAP soil moisture estimate)
- SoilGrids
  - soil_grids_bd_bdod_0-5cm_mean (Soil grids bulk density image)
  - soil_grids_bd_time_millis_diff (SoilGrids bulk density time delta from in-situ label)
  - soil_grids_bd_timestamp (Timestamp of the soil grids bulk density image)
  - soil_grids_clay_clay_0-5cm_mean
  - soil_grids_clay_time_millis_diff (Soil grids clay time delta from in-situ label)
  - soil_grids_clay_timestamp (Timestamp of the soil grids clay image)
  - soil_grids_sand_sand_0-5cm_mean
  - soil_grids_sand_time_millis_diff (Soil grids sand time delta from in-situ label)
  - soil_grids_sand_timestamp (Timestamp of the soil grids sand image)
  - soil_grids_silt_silt_0-5cm_mean
  - soil_grids_silt_time_millis_diff (Soil grids silt time delta from in-situ label)
  - soil_grids_silt_timestamp (Timestamp of the soil grids silt image)
- Soil moisture
  - sm_0_5 (Soil moisture reading at 5cm from non ISMN in-situ sensors. The field is not present for ISMN based data.)

The sm_0_5 field is only present for the SMAP validation sensors (except SMAP-REMEDHUS, SMAP-TONZIRANCH and SMAP-TWENTE since we did not have redistribution permissions for them). For the ISMN data, we provide a script that the user can use to merge ISMN data that they download with the data we provide. More details in the next section.

### Script to combine ISMN soil moisture readings

In order to combine ISMN in-situ readings with the partial dataset provided along with the paper, we provide the script - "combine_ismn_data.py". Users can run this script in order to generate the final full dataset and save it to a destination of their choice.

- Follow the steps below to download the ISMN data
  - Visit https://ismn.geo.tuwien.ac.at/.
  - Login to/create an ISMN account.
  - Navigate to "Data".
  - Download data from all the sensors with a time range of 2015-01-01 to 2022-01-01.
  - Download data in the CEOP format.
  - This will result in a zip file with the ISMN data and should be ~4GB in size.

```code
python3 -m combine_ismn_data --ismn_data_path='<PATH_TO_DOWNLOADED_ISMN_DATA_ZIP>' --output_dir='<PATH_TO_STORE_FULL_DATASET>' --num_workers=16
```

Note: num_workers can be adjusted as required. Having a larger pool of workers will allow for faster processing but depending on your system, it might hang up if the workers are too large. A good range to try would be somewhere within [8, 64].

Note: Once the script is done executing, it will print out success, lookup_failure and value_failure counters. There should be ~110k successes, no value failures, < 750 (~0.75%) lookup failures (happens when ISMN changes data for certain sensors etc).

## Pre-trained Models

We release our best performing model along with the paper. This model uses Sentinel-1, Sentinel-2, NASA DEM, SoilGrids, SMAP and GLDAS inputs to produce the soil moisture estimate.

The model is available for download at https://storage.googleapis.com/eoscience-public/soil_moisture_retrieval_model/soil_moisture_model.zip

### Script to run sample inference

Users can run sample inference on the model we provide along with the dataset we provide (either partial or full, since labels are not required for inference).

```code
python3 -m run_inference --input_data_path='<PATH_TO_A_TFRECORD.GZ_FILE>' --exported_model_dir='<PATH_TO_THE_DOWNLOADED_SOIL_MOISTURE_MODEL>' --num_inference_samples=20
```

Note: exported_model_dir should point to the root directory of the downloaded model which contains 'graph_def.pbtxt', 'saved_model.pb' and a variables directory.

## Results

A comparison of our model performance vs baselines on the soil moisture data:

|                                                          |     | Validation  | |    | Test |  |
| -------------------------------------------------------- | -------------- | -------------- | ----------- | ------ | ------- | ----------- |
| **Experiment**                                               | **ubRMSE**         | **RMSE**           | **Correlation** | **ubRMSE** | **RMSE** | **Correlation** |
| **Baselines (Low-resolution)**                               |
| SMAP                                                     | 0.097          | 0.144          | 0.638 | 0.1 | 0.134 | 0.638 |
| GLDAS                                                    | 0.07           | 0.114          | 0.572 | 0.07 | 0.11 | 0.58 |
| SMAP + GLDAS NN                                          | 0.061          | 0.102          | 0.663 | 0.063 | 0.099 | 0.668 |
| **Ours (High-resolution)**                                  |
| Sentinel-1 + DEM                                         | 0.073          | 0.099          | 0.474 | 0.075 | 0.109 | 0.437 |
| Sentinel-1 + DEM + Sentinel-2                            | 0.067          | 0.094          | 0.587 | 0.069 | 0.099 | 0.56 |
| Sentinel-1 + DEM + Sentinel-2 + SoilGrids                | 0.058 (+4.9%)  | 0.089 (+12.7%) | 0.675 (+1.8%) | 0.06 (+4.7%) | 0.096 (+3%) | 0.647 (-3.1%) |
| Sentinel-1 + DEM + Sentinel-2 + SoilGrids + SMAP + GLDAS | 0.054 (+11.5%) | 0.085 (+16.7%) | 0.727 (+9.7%) | 0.055 (+12.7%) | 0.088 (+11.1%) | 0.729 (+9.1%) |

Validation and test results. A ‘+’ in percentage change denotes an increase in correlation (decrease in ubRMSE) relative to the SMAP + GLDAS NN baseline. Please refer to the paper for additional results.

## Contributing

Please feel free to reach out to us if you'd like to contribute to the code here or if you have any questions in general about the code and how to use it.
