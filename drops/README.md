**Experimental code for DROPS projects on class-imbalanced learning tasks**

**Example Usage**

(1) Two-step vatriant for training DROPS

Step 1: run synthetic long-tailed cifar experiments with ce loss to get the base model

```
python -m drops.main_lt --dataset 'cifar10' --loss 'ce' --imb_ratio 0.1 --dro_div 'kl'
```

Step 2: perform Distribution RObust PoSthoc on saved model prediction logits obtained from the pre-trained model
```
python -m drops.drops_test_time --dataset 'cifar10' --loss 'drops' --imb_ratio 0.1 --eta_lambda 10 --num_iters 25 --eta_lambda_mult 0.95 --prior_type 'train' --cal_type 'none' --eps 0.9
```

(2) One-step vatriant for training DROPS

To run synthetic long-tailed cifar experiments with ce loss, run

```
python -m drops.main_lt --dataset 'cifar10' --loss 'drops' --imb_ratio 0.1 --dro_div 'kl'
```
