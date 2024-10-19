### Generalization and Learnability in Multiple Instance Regression

This repository contains the code for the paper:

**Generalization and Learnability in Multiple Instance Regression** <br>
Kushal Chauhan, Rishi Saket, Lorne Applebaum, Ashwinkumar Badanidiyuru Varadaraja, Chandan Giri, Aravindan Raghuveer <br>
*Uncertainty in Artificial Intelligence (UAI 2024)*

#### Environment setup
1. Create a new conda environment `conda create -n mir_uai24 python=3.11`.
2. Activate it `conda activate mir_uai24`.
3. Install the requirements `pip install -r mir_uai24/requirements.txt`

#### Data preparation
1. Prepare synthetic data for different bag sizes `python -m mir_uai24.dataset_preparation.prep_synthetic --bag_size <bag_size>`
2. Download 1940 US Census data from [here](https://usa.ipums.org/usa/1940CensusDASTestData.shtml) as a csv file.
3. Prepare US Census data for different bag sizes `python -m mir_uai24.dataset_preparation.prep_us_census --read_path <downloaded_csv_path> --bag_size <bag_size>`

#### Usage
The `train.py` script will train an MLP using the proposed **wtd-Assign** approach on the desired dataset. To reproduce results on the synthetic data, run

```
python -m mir_uai24.train --dataset synthetic --batch_size 100 --embedding_dim 0 --num_hidden_units 1024
```

To reproduce results on the 1940 US Census data, run

```
python -m mir_uai24.train --dataset us_census --batch_size 100 --embedding_dim 0 --num_hidden_units 1024
```


#### Citation
If you find our methods useful, please cite:

```
@InProceedings{chauhan-uai24,
    title     = {Generalization and Learnability in Multiple Instance Regression},
    author    = {Kushal Chauhan and Rishi Saket and Lorne Applebaum and Ashwinkumar Badanidiyuru Varadaraja and Chandan Giri and Aravindan Raghuveer},
    booktitle = {Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence},
    month     = {July},
    year      = {2024}
}
```>>>>>>> source:           1f98f9da787d - kushalchauhan: PUBLIC: Open-sourcing...
