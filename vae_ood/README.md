### Robust outlier detection by de-biasing VAE likelihoods

This repository contains the code for the paper:

**Robust outlier detection by de-biasing VAE likelihoods** <br>
Kushal Chauhan, Barath Mohan Umapathi, Pradeep Shenoy, Manish Gupta, Devarajan Sridharan <br>
*IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022)* <br>


#### Preparing
1. Creare a new conda environment `conda create -n vae_ood python=3.7.10`.
2. Activate it `conda activate vae_ood`.
3. Install the requirements `pip install -r vae_ood/requirements.txt`
2. SignLang, CompCars and GTSRB datasets have to be downloaded manually. Download the SignLang dataset from [Kaggle](https://www.kaggle.com/ash2703/handsignimages) (requires Kaggle account) and extract the archive contents in `vae_ood/datasets/SignLang`. For CompCars surveillance data, download `sv_data.zip`  from [here](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/) (scroll down for download instuctions)  and extract the archive contents in `vae_ood/datasets/sv_data`. For GTSRB, download `GTSRB_Final_Training_Images.zip` and `GTSRB_Final_Test_Images.zip` from [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html) and extract the contents in `vae_ood/datasets/GTSRB`.

#### Usage

The `main.py` script will train a VAE for a desired dataset and compute LL and (CS+)BC-LL for all OOD datasets.

To reproduce the results for the Fashion-MNIST VAE, run

```
python -m vae_ood.main --dataset fashion_mnist --visible_dist cont_bernoulli --do_train --do_eval --latent_dim 20 --num_filters 32 --experiment_dir vae_ood/models/cont_bernoulli
```


To reproduce the results for the CelebA VAE, run

```
python -m vae_ood.main --dataset celeb_a --visible_dist cont_bernoulli --do_train --do_eval --latent_dim 20 --num_filters 64 --experiment_dir vae_ood/models/cont_bernoulli
```

The `probs.pkl` files written by `main.py` can be used to get AUROC/AUPRC/FPR@80 metrics reported in the paper as shown in `results.ipynb`.

#### Citation
If you find our methods useful, please cite:

```
@InProceedings{Chauhan_2022_CVPR,
    author    = {Kushal Chauhan and Barath Mohan Umapathi and Pradeep Shenoy and Manish Gupta and Devarajan Sridharan},
    title     = {Robust outlier detection by de-biasing VAE likelihoods},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022}
}
```
