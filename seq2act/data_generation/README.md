## Generate RicoSCA Datasets

**Download Rico Public Dataset**
```
# Download dataset from http://interactionmining.org/rico#quick-downloads
# Choose 1 UI Screenshots and View Hierarchies (6 GB)
# Place the downloaded Rico .json data under folder seq2act/data/rico_sca/raw

Create a folder named "output" under "seq2act/data/rico_sca"
```
**Generate Rico SCA tfrecord**
```
sh seq2act/data_generation/create_rico_sca.sh
```
