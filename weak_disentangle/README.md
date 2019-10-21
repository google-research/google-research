# Weakly Supervised Disentanglement with Guarantees

This codebase trains the models analyzed in the paper: Weakly Supervised Disentanglement with Guarantees.

In addition to installing the requirements, make sure to follow the instructions from [disentanglement_lib](https://github.com/google-research/disentanglement_lib) to download the necessary datasets in the parent directory that contains the folder `weak_disentangle`.

To run this code with a choice of default config (for complete-change-pairing on dsprites) from the parent directory, simply run
```
python3 -m weak_disentangle.main
```

### Changing the Configuration

The codebase uses gin-config to specify the model being run. A full default specification is provided in `weak_disentangle/configs/gan.gin`. Below, we explain the behavior of the important configuration parameters:

```
train.model_type: There are two model types:
  - "gen": specifies the use of a paired GAN (for match pairing and rank pairing).
  - "van": specifies use of a vanilla GAN (for restricted labeling and full labeling).

train.dset_name: We used five datasets in our experiments:
  - "shapes3d"
  - "dsprites"
  - "scream"
  - "norb"
  - "cars3d"

train.s_dim: The number of true underlying factors:
  - 6: for "shapes3d"
  - 5: for "dsprites" and "scream"
  - 4: for "norb"
  - 3: for "cars3d"

train.n_dim: The number of nuisance factors:
  - 0: for "shapes3d", "dsprites", "cars3d"
  - 2: for "scream", "norb"

train.factors: The supervision procedure. Here are some examples:
  - "s=0,1,2,3,4": share-labeling on factors 0,1,2,3,4 individually
  - "c=0,1,2,3,4": change-labeling on factors 0,1,2,3,4 individually
  - "c=012,234": change-labeling on the group of factors {0,1,2} and the group of factors
    {2,3,4}. The intersection rule will result in restrictiveness on factor 2.
  - "r=0,1": ranking on factors 0,1 individually
  - "l=": vanilla GAN with no labels
  - "l=1,2": vanilla GAN with restricted labeling on factors 1 and 2 individually.
  - "cs=0,1": both change and share pairing on factors 0,1 individually
  These examples should hopefully give you a sense of how to set train.factors for your desired setting.

mask_type: There are three mask types:
  - "match"
  - "label"
  - "rank"
  You need to make sure that the mask type match your train.model_type and train.factors choice
```

In our code, we also have five architectural hyperparameters that we modified:
```
initializer.method:
  - "keras"
  - "pytorch"

train.enc_lr_mul:
  - 1
  - 2

Discriminator.width:
  - 1
  - 2

Discriminator.share_dense:
  - True
  - False

Discriminator.uncond_bias:
  - True
  - False
```

When using vanilla GAN, the Discriminator is swapped out for the LabelDiscriminator:
```
LabelDiscriminator.width:
  - 1
  - 2

LabelDiscriminator.share_dense:
  - True
  - False

LabelDiscriminator.uncond_bias:
  - True
  - False
```

If you have any questions about the project, reach out to Rui on Twitter [@_smileyball](https://twitter.com/_smileyball) or contact via email.
