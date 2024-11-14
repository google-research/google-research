# MAGNET: MAGnitude Neural EsTimation model

This code implements MAGNET for magnitude prediction of earthquakes.
MAGNET is presented in the paper _Do earthquakes "know" how big they will be? a neural-net aided study_ https://doi.org/10.48550/arXiv.2408.02129 .

# Install

The package was tested using a python env created with conda package manager.
From the root of project, perform:

```
conda create --name <environment_name> --file requirements.txt -c conda-forge 
conda activate <environment_name> 
pip install --no-deps -e .
```

# Quick start
A tutorial for this package can be seen in the ```tutorial_run_this_first.ipynb``` notebook.

# Model training procedure

The general scheme of training a model using MAGNET is as follows:

```mermaid
graph LR
A([Compute features])  --> B(Train model) --> C(Examine results)
D([Compute benchmarks]) --> C
```

_Compute features_ and _Compute benchmarks_ stages are performed separately from
the training process, results are cached and read during training or when
needed.

_Compute features,  Compute benchmarks_ and _Train model_ are executed by
scripts (see package structure below). 

A few examples of how to _Examine results_ are given in the `notebooks` directory.

# Package structure

Below is the detailed folder content

```
eq_mag_prediction
├── eq_mag_prediction # package code
│   ├── scripts       # meant for running from command line interface
│   ├── ingestion     # ingestion of catalogs
|   :
|
├── notebooks         # ipynb notebooks exemplifying functionality
└── results           # meant for storage of script results and pre trained models
    ├── cached_benchmarks
    ├── cached_features
    ├── catalogs
    │   ├── ingested
    │   └── raw
    └── trained_models
        ├── GeoNet_NZ
        ├── Hauksson
        └── JMA

```

## notebooks
ipynb notebooks exemplifying usage of model loading and result analysis.


## scripts

Scripts are meant for execution by CLI, after activating the workspace:

1.  `calculate_benchmark_gr_properties.py`: calculates and caches benchmarks for
    comparison.
2.  `magnitude_prediction_compute_features.py`: computes features for model
    training.
3.  `magnitude_predictor_trainer.py`: train a model.

By running
```
python3  path/to/script.py --help
```
A detailed description of the script and flags to be used will be presented.

> <span style="font-size:0.8em;"> Cached results from
> `calculate_benchmark_gr_properties.py` and
> `magnitude_prediction_compute_features.py` are named using hash code that
> indicates the exact inputs that were used to calculate them. The names
> themselves are not fully indicative, and are read using dedicated reading
> functions that search for the specific encoded names. </span>

### gin configurations
Running scripts require a ```config.gin``` file. These files define to specific parameters to be used in the executed run. These may include what catalog is used, the training period defined, hyperparameters for model training, etc. 
Some scripts are directed to a default gin configuration file.

For more details on how to use gin see:
https://github.com/google/gin-config?tab=readme-ov-file

## ingestion (a.k.a "Work on you own catalog")

In order to use a catalog with MAGNET it should be transformed from its 
raw format  into a standard format, this is the ***ingestion*** process.

The standard form which can be used by MAGNET can be seen in the mock catalog produced by the tutorial ```tutorial_run_this_first.ipynb```. 

Ingestion scripts for the catalogs we have worked on can be found in this code base, and are meant to be run via the command line interface. 
By default, ingestion scripts will look for the raw catalog in
`eq_mag_prediction/results/catalogs/raw` and cache the modified in
`eq_mag_prediction/results/catalogs/ingested`. In order to change this behavior
see flags in ingestion code itself.

> JMA catalog may be also ingested via a protobuf format (https://protobuf.dev/)
>instead of pythonically. See description of ```ingest_jma_via_proto.py``` for
>details.

> ```pdfs_of_major_earthquakes.ipynb``` plots model's prediction for a list of
major earthquakes in the catalog region. By the notebook design,  the list of
major earthquakes are given by the ingested major earthquakes catalog, per
region. See description in ```ingest_***_major_earthquakes.py``` scripts.

For technical reasons, we cannot include in this code base the catalogs we have analysed themselves. You can download the catalogs from the following links and ingest them with the relevant script:



* Hauksson catalog for Southern California:<br> https://service.scedc.caltech.edu/ftp/catalogs/hauksson/Socal_focal/

*  JMA catalog for Japan:<br> 
https://www.data.jma.go.jp/svd/eqev/data/bulletin/hypo_e.html
* The New Zealand GeoNet catalog:<br> 
https://www.geonet.org.nz/data/types/eq_catalogue

## results
Where trained models are stored, and cached by default.

Default cache location for benchmarks, features and ingested catalogs as well.

Pre-trained models, encoders, and calculated benchmarks can be found in: https://zenodo.org/records/13387662 . The mentioned zenodo repository contains a full "results" directory.