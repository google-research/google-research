# Implicit Representation of Probability Distributions on the Rotation Manifold

<img src="https://implicit-pdf.github.io/ipdf_files/cube.gif" width="800"/>

[Project Page](https://implicit-pdf.github.io/) |
[Paper](arxiv.org)

This code accompanies the paper

**Implicit Representation of Probability Distributions on the Rotation Manifold**

Kieran Murphy*, Carlos Esteves*, Varun Jampani, Srikumar Ramalingam, Ameesh Makadia

ICML 2021 ([arxiv]())


## Repository Contents

## Python environment
The code was run successfully with Python 3.6.12.  The necessary libraries may
be installed using pip with the following line:

`pip install -r implicit_pdf/requirements.txt`

## Training a pose estimator from scratch

The following example call

`python -m implicit_pdf.train --symsol_shapes cube --so3_sampling_mode random --num_fourier_components 4`

trains an IPDF pose estimator from scratch on the cube from the SYMSOL I dataset, with four positional encoding frequencies on the rotation query.

## Generating an equivolumetric grid on SO(3)

Evaluation requires exact normalization of the predicted distributions.
To this end we use an equivolumetric grid covering SO(3).
Following [Yershova et al. (2010)](http://lavalle.pl/papers/YerLavMit08.pdf), we include the necessary code to produce grids of different sizes using the [HealPix](https://healpix.jpl.nasa.gov/) method as a starting point.
The [HealPy library](https://healpy.readthedocs.io/en/latest/) is required and is installed with the above `pip` call with `requirements.txt`.
The grids are automatically created as needed through the training script but may also be used in a standalone fashion with `generate_healpix_grid()` in `implicit_pdf/models.py`.

During training, a simpler mode of querying SO(3) -- sampling uniformly at random -- is also effective, even though normalization is no longer exact.
This can be set via `--so3_sampling_mode=random` in the above call, where the number of queries during training may be specified exactly.

## SYMSOL dataset
Accompanying this code release, the symmetric solid (SYMSOL) datasets introduced with the paper have been added to tensorflow_datasets.
At present, it can be accessed through [tfds-nightly](https://www.tensorflow.org/datasets/catalog/symmetric_solids).

There are 50,000 renderings each of eight shapes (five featureless from SYMSOL I and three marked shapes from SYMSOL II).
Each 224x224 RGB image is accompanied by the single ground truth rotation of the camera
during rendering, as well as the full set of equivalent ground truths under symmetry (discretized at 1 degree intervals for the cone and cylinder) for the five shapes of SYMSOL I.
![dataset](https://implicit-pdf.github.io/ipdf_files/symsol_dataset.gif)

### Citation

If you found this work or dataset useful, please cite

```
@inproceedings{implicitpdf2021,
  title = {Implicit Representation of Probability Distributions on the Rotation Manifold},
  author = {Murphy, Kieran and Esteves, Carlos and Jampani, Varun and Ramalingam, Srikumar and Makadia, Ameesh}
  booktitle = {International Conference on Machine Learning}
  year = {2021}
}
```
