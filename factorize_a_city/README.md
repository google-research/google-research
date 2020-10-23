# Learning to Factorize and Relight a City
[Project Page](https://factorize-a-city.github.io/) |
[Paper](https://arxiv.org/abs/2008.02796) 

This code accompanies the paper

**Learning to Factorize and Relight a City** \
Andrew Liu, Shiry Ginosar, Tinghui Zhou, Alexei A. Efros, Noah Snavely \
ECCV 2020

*Please note that this is not an officially supported Google product.*

## Python Environment

The following code base was successfully ran with Python 3.7.9. We suggest
installing the library in a vitual environment as our code requires older
versions of libraries.

To install using pip, run: \
`pip3 install -r requirements.txt`

## Data

We include a small sample of our NYC test stack to test our code on. To access
the resources, run `source download_sample_resources.sh`.

## Scripts

#### Alignment

In order to align new stacks, we run twenty gradient descent steps of alignment
optimization with frozen network weights. For the test stacks provided, we have
already computed their alignment with this process and saved the results in
`alignment.npy`.

Command: \
`python -m factorize_a_city.align_stack --misaligned_stack_folder=factorize_a_city/data/000057`

#### Decomposition

Recovers the intrinsic image components (log reflectance and log shading) from
an input stack of panoramas.

Command: \
`python -m factorize_a_city.compute_intrinsic_components --stack_folder=factorize_a_city/data/000057
--output_dir=factorize_a_city/intrinsic_image_results`

#### Sun Position Relighting

Given an input stack representing the same scene and a desired lighting context
specified from `data/lighting_context.npy` by `lighting_context_index`,
generates a sequence of sun positions around the entire input scene.

Command: \
`python -m factorize_a_city.rotate_sun_azimuth --stack_folder=factorize_a_city/data/000057
--lighting_context_index=1 --azimuth_frame_rate=10 --output_dir=factorize_a_city/rotate_results`

#### Lighting Condition Relighting

Relights an input panorama stack using illumination conditions copied from
exemplar test panoramas. These factors are saved in `factorize_a_city/data/azimuth.npy` and
`factorize_a_city/data/lighting_context`.

Command: \
`python -m factorize_a_city.relight_scene --stack_folder=factorize_a_city/data/000057 --output_dir=factorize_a_city/relit_results`
