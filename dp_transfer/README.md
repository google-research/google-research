# Large Scale Transfer Learning for Differential Privacy 

We empirically illustrate that transfer learning is an extremely effective
technique for training high-utility and large classification models with
differential privacy constraints. In fact, we show that it is indeed possible
to achieve a new state of the art and obtain ~88% top-1 accuracy on challenging
ImageNet-1k benchmark with $$(\epsilon=8, \delta=8e-7)$$ privacy constraints
by employing our proposed recommendations.

More details can be found in https://arxiv.org/abs/2205.02973 and 
https://arxiv.org/abs/2211.13403.

# Usage

We provide 4 methods and DP sanitization mechanisms for private finetuning using
pre-trained features.

Namely,

* ```DP-Adam```
* ```DP-Newton```
* DP-Least Squares (```DP-LS```) or DP-Linear Regression (```DP-LR```)
* DP-SGD with Feature Covariance (```DP-FC```)


These can be configured and used using config files found in the ```configs``` dir.
We configure datasets using separate set of configs found in the ```data_configs```
dir.

The code is meant to have self-contained privacy sanitizers and trainers for
all 4 methods, so that they can be easily imported or used in another
codebase. Even though, most of the code can be used as a reference for your own 
implementation, we also provide a binary which can be used to run our methods. 

For instance, following command can be used for DP-FC:

```sh
python main.py \
--config=experimental/users/harshm/dp/configs/fc_regression.py \
--workdir=/tmp/dp_fc
```

The computational constraints are not much since we are only 
training the last layer, although we still recommend the use of a GPU or TPU for 
training.

## Citation
If you find our code or ideas useful, please cite:

```
@article{Mehta2022LargeST,
  title={Large Scale Transfer Learning for Differentially Private Image Classification},
  author={Harsh Mehta and Abhradeep Thakurta and Alexey Kurakin and Ashok Cutkosky},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.02973}
}
```

and 

```
@article{Mehta2022DifferentiallyPI,
  title={Differentially Private Image Classification from Features},
  author={Harsh Mehta and Walid Krichene and Abhradeep Thakurta and Alexey Kurakin and Ashok Cutkosky},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.13403}
}
```

## License

Licensed under the Apache 2.0 License.

## Disclaimer

This is not an officially supported Google product.