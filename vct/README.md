# VCT: A Video Compression Transformer

Fabian Mentzer, George Toderici, David Minnen, Sung-Jin Wang, Sergi Caelles, Mario Lucic, Eirikur Agustsson

https://arxiv.org/abs/2206.07307

To appear at NeurIPS'22.



## Abstract

We show how transformers can be used to vastly simplify neural video
compression. Previous methods have been relying on an increasing number of
architectural biases and priors, including motion prediction and warping
operations, resulting in complex models. Instead, we independently map input
frames to representations and use a transformer to model their dependencies,
letting it predict the distribution of future representations given the past.
The resulting video compression transformer outperforms previous methods on
standard video compression data sets. Experiments on synthetic data show that
our model learns to handle complex motion patterns such as panning, blurring and
fading purely from data. Our approach is easy to implement, and we release code
to facilitate future research.


## Code

**Work In Progress, Code Coming Soon**


## Data from Fig. 4


<!--
-->

Data for all curves labeled `VCT (Ours)` for all four subplots.

### MCL-JCV, PSNR

```csv
bpp,PSNR
0.02085819604853945,32.80706320932177
0.0326342776907401,34.204882345687004
0.04992265997430723,35.57629071470896
0.07706905056134696,36.79151286947463
0.1244432485481983,37.91397499669394
0.2053834664609076,38.95594312074448
0.3246340299789379,39.87937358864677
0.518594972272714,40.85205932227242
```

### UVG, PSNR

```csv
bpp,PSNR
0.013792417188551986,32.487814650308515
0.022362519427357862,33.931710855393185
0.0349990398529917,35.267389486403694
0.054894117672617246,36.43507358278548
0.08854167145100376,37.51903957003639
0.15123796656373,38.49818090529669
0.26794948341945807,39.37668041320074
0.4936706910424289,40.42103525797527
```

### MCL-JCV, MS-SSIM

```csv
bpp,MS-SSIM
0.014692373034541703,0.9417002881619668
0.025895096142646963,0.9597873670127655
0.04478931982359953,0.9707673285735978
0.0846302819899749,0.9792074928416149
0.1488331270555035,0.9844152062217394
0.2603574583977047,0.9886679214616616
0.445014919663469,0.9919034840775861
0.699143494782514,0.9941178404053053
```

### UVG, MS-SSIM

```csv
bpp,MS-SSIM
0.012952257920211803,0.9304468844901947
0.0230215588440409,0.950741335139388
0.040960844996873115,0.9637830031343867
0.07894400571073804,0.9738082034247262
0.14644523799419404,0.9804844583641915
0.26371675850380033,0.9857407963701658
0.47123691950525554,0.989954089891343
```
