
This repository contains reference code for the ICLR-23 paper "[KwikBucks: Correlation Clustering with
Cheap-Weak and Expensive-Strong signals](https://openreview.net/forum?id=p0JSSa1AuV)". The code implements the main KwikBucks algorithm of the paper (it is implemented as `qwick_cluster_using_ordering` in `model_utils.py`) as well as the baselines stated in the experimental section of the paper. 

## Datasets
The datasets can be downlowded from here: https://storage.googleapis.com/gresearch/kwikbucks/kwikbucks.zip

## Installation
* Download the code. 
* Download the the dataset and put it under `kwikbucks/data/`.
* Run `run.sh` to install `requirements.txt` and run the code.


## Citation
If you use the code, please cite our paper.
> Silwal, S., Ahmadian, S., Nystrom, A., McCallum, A., Ramachandran, D., Kazemi, M.,
> *"KwikBucks: Correlation Clustering with Cheap-Weak and Expensive-Strong Signals"*,
> The Eleventh International Conference on Learning Representations (ICLR), 2023.
```
@inproceedings{silwal2023kwikbucks,
  title={KwikBucks: Correlation Clustering with Cheap-Weak and Expensive-Strong Signals},
  author={Silwal, Sandeep and Ahmadian, Sara and Nystrom, Andrew and McCallum, Andrew and Ramachandran, Deepak and Kazemi, Seyed Mehran},
  booktitle={The Eleventh International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

## Contact us
For questions or comments about the implementation, please contact mehrankazemi@google.com or silwal@mit.edu.

## Disclaimer
This is not an official Google product.