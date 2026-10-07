# Reversible Data Attribution & Cleansing Pipeline

*This is not an officially supported Google product.*

<!-- disableFinding("Data cleansing") -->
<!-- disableFinding("data cleansing") -->

This repository contains the implementation of reversible data attribution and
memory-efficient influence functions for neural networks trained with Adam and
SGD.

---

## 1. Overview & Concepts

Data cleansing evaluates the quality of influence and data attribution methods
by identifying and removing noisy or mislabeled training points (e.g., corrupted
samples), retraining the model without those points, and measuring downstream
test performance (loss reduction and accuracy improvement).

While standard data cleansing operates **offline** (scoring samples post-hoc
after full model training), this codebase also contains an **online pruning**
framework (see
[Section 5: Online Data Pruning Framework](#5-online-data-pruning-framework-next-steps))
that identifies and masks harmful points progressively during model training.

### Key Attribution Methods

| Method Category | Method Identifiers | Computational Complexity | Description |
| :--- | :--- | :--- | :--- |
| **Fast Recursive / Closed-form** | `adam_recursive`, `adam_recursive_nodv` | $O(T)$ backpropagation | Second-order Hessian-vector product backward recursion for Adam. Fast & computes all sample scores simultaneously. |
| **First-Order Checkpoint** | `tracin_adam`, `tracin_sgd` | $O(C \cdot N)$ | TracIn gradient-dot-product attribution across stored training checkpoints. |
| **Classical Influence** | `icml` | $O(N)$ (final checkpoint) | Koh & Liang (ICML 2017) influence function approximation using Hessian inverse on the final model. |
| **Outlier & Baseline** | `ae`, `iso`, `random` | Fast ($O(N)$) | Unsupervised baseline methods: Autoencoder reconstruction error (`ae`), Isolation Forest (`iso`), and random scoring (`random`). |
| **Slow Forward / Trajectory Tracking** | `adam_exact` (`adam_forward`), `sgd_all`, `sgd_last` | $O(N \cdot T)$ forward passes | Forward tracking of parameter, momentum, and variance perturbations for individual candidate samples. |
| **Ground Truth Counterfactuals** | `--eval_counterfactuals=True` | $O(N \cdot \text{train})$ | Exact Leave-One-Out (LOO) retraining for every candidate sample index. |

### Naming & Implementation Notes

-   **TSLOO Adam (our method)**: `adam_recursive` is `TSLOO-Adam` but with
    recursion done in backwards manner (with first-order approximation in terms
    of sample-wise gradient).
-   **Influence function**: `'icml'` should really be `'influence function'`.

---

## 2. Recommended Workflows

### A. Fast Methods Workflow (Single-Run / `mode=all`)

For fast methods (`adam_recursive`, `tracin_adam`, `tracin_sgd`, `icml`, `ae`, `iso`, `random`), score computation runs in a single fast pass. You can run the entire pipeline end-to-end at once.

#### Example (MNIST):

```bash
python -m reversible_data_attribution.run_data_cleansing_eval \
  --mode=all \
  --dataset=mnist \
  --model_type=cnn \
  --optimizer_type=adam \
  --num_train=50000 \
  --num_val=10000 \
  --num_test=10000 \
  --methods=adam_recursive,tracin_adam,ae,iso,random \
  --noise_type=label_flip \
  --noise_rate=0.1 \
  --num_epochs=10 \
  --k_list=0,1,5,10,25,50,100 \
  --auto_cleansing=True \
  --output_dir=/tmp/cleansing_fast_run
```

#### Example (Synthetic Dataset):

```bash
python -m reversible_data_attribution.run_data_cleansing_eval \
  --mode=all \
  --dataset=synthetic \
  --model_type=dnn \
  --optimizer_type=adam \
  --num_train=1000 \
  --num_val=200 \
  --num_test=200 \
  --methods=adam_recursive,tracin_adam,random \
  --num_epochs=5 \
  --output_dir=/tmp/cleansing_fast_run
```

> [!IMPORTANT] **Passing Sample Counts is Necessary**: It is **necessary to pass
> in the number of samples (`--num_train`, `--num_val`, and `--num_test`)**. If
> omitted, the pipeline falls back to small built-in defaults (1000 training,
> 200 validation, and 200 test samples) instead of the full dataset (e.g.,
> 50,000 train, 10,000 val, 10,000 test for MNIST). Always explicitly specify
> these flags for your experiments.

Note: we only do `--optimizer_type=adam`, as the goal is to analyze the
behaviour of various data attribution methods under a fixed training procedure
(Adam optimizer).

We detail some of the successful runs below:

For example, here are some of the successful runs:

```
 --dataset=mnist \
 --model_type=dnn \
 --methods=adam_recursive \
 --noise_type=label_flip \
 --noise_rate=0.1 \
 --num_epochs=50 \
 --num_train=50000 \
 --num_val=10000 \
 --num_test=10000 \
 --learning_rate=2e-5 \
 --auto_cleansing=True
```

#### Flag Overview:

| Flag               | Example Value             | Description                 |
| :----------------- | :------------------------ | :-------------------------- |
| `--mode`           | `all`                     | Execution phase: runs the   |
:                    :                           : complete pipeline (train    :
:                    :                           : base model, compute         :
:                    :                           : attribution scores, cleanse :
:                    :                           : data, and retrain).         :
| `--dataset`        | `mnist`                   | Dataset used for training   |
:                    :                           : and evaluation (`mnist`,    :
:                    :                           : `cifar10`, `cifar100`,      :
:                    :                           : `fashion_mnist`, `imdb`).   :
| `--num_train`      | `50000`                   | Number of training samples  |
:                    :                           : (**necessary to pass in**;  :
:                    :                           : default `1000` is a small   :
:                    :                           : toy subset).                :
| `--num_val`        | `10000`                   | Number of validation        |
:                    :                           : samples used for query      :
:                    :                           : gradients (**necessary to   :
:                    :                           : pass in**; default is       :
:                    :                           : `200`).                     :
| `--num_test`       | `10000`                   | Number of test samples for  |
:                    :                           : evaluation (**necessary to  :
:                    :                           : pass in**; default is       :
:                    :                           : `200`).                     :
| `--model_type`     | `cnn`                     | Neural network architecture |
:                    :                           : (`cnn`, `resnet`,           :
:                    :                           : `resnet18`, `small_dnn`,    :
:                    :                           : `dnn`, `vit`, `linear`).    :
| `--optimizer_type` | `adam`                    | Optimization algorithm for  |
:                    :                           : model training and          :
:                    :                           : retraining (`adam` or       :
:                    :                           : `sgd`).                     :
| `--methods`        | `adam_recursive,...`      | Comma-separated list of     |
:                    :                           : attribution methods to      :
:                    :                           : evaluate.                   :
| `--noise_type`     | `label_flip`              | Synthetic noise type        |
:                    :                           : applied to corrupt training :
:                    :                           : data (`label_flip`,         :
:                    :                           : `gaussian`, `lowpass`,      :
:                    :                           : `highpass`, `both`,         :
:                    :                           : `none`).                    :
| `--noise_rate`     | `0.1`                     | Fraction of training        |
:                    :                           : samples to corrupt ($0.1 =  :
:                    :                           : 10\%$).                     :
| `--num_epochs`     | `10`                      | Number of training epochs   |
:                    :                           : for initial model training  :
:                    :                           : and counterfactual          :
:                    :                           : retraining.                 :
| `--k_list`         | `0,1,5,10,25,50,100`      | List of removal counts $k$  |
:                    :                           : for evaluating top-$k$      :
:                    :                           : suspicious sample           :
:                    :                           : cleansing.                  :
| `--auto_cleansing` | `True`                    | Automatically identifies    |
:                    :                           : and removes all samples     :
:                    :                           : whose predicted influence   :
:                    :                           : reduces validation loss     :
:                    :                           : ($\Delta L < 0$).           :
| `--output_dir`     | `/tmp/cleansing_fast_run` | Destination directory where |
:                    :                           : evaluation CSVs,            :
:                    :                           : `scores.npz`, and           :
:                    :                           : `config.json` are written.  :

*(For the complete parameter list and advanced flags, see
[Section 3: Parameter Reference](#3-parameter-reference-run_data_cleansing_evalpy).)*

#### What happens in `mode=all`:

1. **Full Model Training**: Base model trains on the corrupted dataset; checkpoints, momentum, and variance trajectories are tracked.
2. **Score Computation**: Scores are computed across all specified methods.
3. **Score Saving**: Saved to `scores.npz` (and `corrupted_indices.json`).
4. **Data Cleansing & Retraining**: Top-$k$ suspicious samples (and automatically identified loss-reducing samples) are removed, the model is retrained, and test metrics are recorded.

---

### B. Slower / Index-Splittable Methods Workflow (Distributed Map-Reduce)

For computationally demanding methods that iterate across individual candidate training indices (`adam_exact`, `sgd_all`, `sgd_last`, or full ground truth counterfactuals with `--eval_counterfactuals=True`), running all samples sequentially on a single worker is too slow.

Use a two-stage **Map-Reduce** workflow:

```
[Stage 1: Map Phase]
  Shard 0: indices [0, 500)   --> scores_shard_0000.npz
  Shard 1: indices [500, 1000) --> scores_shard_0001.npz
  Shard 2: indices [1000, 1500) --> scores_shard_0002.npz
  ...

[Stage 2: Reduce & Cleansing Phase]
  merge_sharded_scores() --> consolidated scores_dict --> automated top-k removal & retraining
```

#### Step 1: Compute Scores (Map Phase — `--mode=compute_scores`)
Partition training samples across shards using `--shard_id` and `--num_shards`.

```bash
# Example: Shard 0 of 4 (via Python)
python -m reversible_data_attribution.run_data_cleansing_eval \
  --mode=compute_scores \
  --dataset=mnist \
  --model_type=cnn \
  --methods=adam_exact \
  --num_train=2000 \
  --num_val=200 \
  --num_test=200 \
  --num_shards=4 \
  --shard_id=0 \
  --max_workers=4 \
  --save_scores_dir=/tmp/sharded_scores/exp1

# Example: Shard 1 of 4 (via Python)
python -m reversible_data_attribution.run_data_cleansing_eval \
  --mode=compute_scores \
  --dataset=mnist \
  --model_type=cnn \
  --methods=adam_exact \
  --num_train=2000 \
  --num_val=200 \
  --num_test=200 \
  --num_shards=4 \
  --shard_id=1 \
  --max_workers=4 \
  --save_scores_dir=/tmp/sharded_scores/exp1
```

*Note: Because random seeds (`--noise_seed`) are deterministic, every shard starts with the exact same initial weights and training trajectory.*

#### Step 2: Cleansing & Evaluation (Reduce Phase — `--mode=cleansing_from_scores`)
Once all shards have completed, point `--precomputed_scores_dir` to the folder containing `scores_shard_*.npz`.

```bash
python -m reversible_data_attribution.run_data_cleansing_eval \
  --mode=cleansing_from_scores \
  --dataset=mnist \
  --model_type=cnn \
  --methods=adam_exact \
  --num_train=2000 \
  --num_val=200 \
  --num_test=200 \
  --precomputed_scores_dir=/tmp/sharded_scores/exp1 \
  --auto_cleansing=True \
  --output_dir=/tmp/cleansing_results/exp1
```

The loader automatically finds all `scores_shard_*.npz` files, stitches together the full score vector, selects corrupted points, retrains the model without them, and generates evaluation tables.

> [!IMPORTANT] **Consistent Sample Counts Across Phases**: Pass explicit
> `--num_train`, `--num_val`, and `--num_test` flags to both the Map phase
> (`--mode=compute_scores`) and Reduce phase (`--mode=cleansing_from_scores`),
> ensuring the values are identical. Omitting them will fall back to defaults
> (1000/200/200), resulting in index mismatches between computed scores and
> loaded datasets.

---

### C. Numerical Stability & Memory Optimization: Checkpointing, Reversible Dynamics & Adaptive Clipping

When computing recursive attribution scores (`adam_recursive`), tracking
parameter and optimizer trajectories across many training steps can consume
substantial memory and pose numerical stability challenges.
`run_data_cleansing_eval.py` and `infl_adam.py` provide complementary memory
management and stabilization strategies:

#### 1. Checkpointing Configurations

-   **All-Step Checkpointing (Default)**:
    -   Enabled when `--checkpoint_freq=None` and `--use_reversible=False`.
    -   Saves model weights, first-moment momentum, and second-moment variance
        at **every training step** (`save_all_ckpts=True`).
    -   **Tradeoff**: Provides exact step-level states without approximation
        error, but requires $O(T \cdot |\theta|)$ memory, which may lead to
        out-of-memory (OOM) issues on longer runs or larger models.
-   **Periodic Checkpointing**:
    -   Enabled by specifying a step interval: `--checkpoint_freq=<N>` (or
        `--checkpoint_frequency=<N>`), e.g., `--checkpoint_freq=100`.
    -   Snapshots model and optimizer states every $N$ steps, significantly
        reducing memory footprint while maintaining intermediate state anchors.

```bash
# Example: Periodic checkpointing every 50 steps
python -m reversible_data_attribution.run_data_cleansing_eval \
  --mode=all \
  --dataset=mnist \
  --model_type=cnn \
  --optimizer_type=adam \
  --num_train=50000 \
  --num_val=10000 \
  --num_test=10000 \
  --methods=adam_recursive \
  --checkpoint_freq=50 \
  --output_dir=/tmp/cleansing_checkpointed
```

> [!WARNING] **Floating-Point Error Amplification**: Periodic checkpointing
> without quantized backward reconstruction (`--use_reversible=True`) can lead
> to amplifying floating-point errors for backward recursion methods (e.g.,
> `infl_adam.recursive_update` and `infl_adam.recursive_update_nodv`).
> Reconstructing uncheckpointed intermediate states backward in standard
> floating point requires iteratively dividing by decay factors $\beta_1,
> \beta_2 < 1$, which exponentially amplifies rounding errors over long
> intervals. If memory reduction is required for recursive methods, using
> quantized backward reconstruction (`--use_reversible=True`) is strongly
> recommended.

#### 2. Reversible Adam Configuration

-   **Quantized Backward Reconstruction**:
    -   Enabled with `--use_reversible=True` (or
        `--use_reversible_transform=True`).
    -   Employs reversible Adam dynamics with fixed-point quantization
        (`--quantization_scale=1000000000000`, default $10^{12}$).
    -   Executes a second training pass (`retrain_with_quantize`) to construct
        GPU reversible transform buffers (`GPUReversibleTransform`) for momentum
        and variance, allowing optimizer trajectories to be accurately
        reconstructed backward in time during recursion.
    -   **Minimal Memory Overhead**: When combined with
        `--checkpoint_freq=None`, only step 0 (initial) and the final step
        checkpoints are stored, reducing checkpoint memory to $O(1)$.
    -   **Hybrid Mode**: Can also be paired with periodic checkpointing (e.g.
        `--use_reversible=True --checkpoint_freq=100`) to store periodic
        checkpoints as verification checkpoints.
    -   **Drift Logging**: Automatically logs numerical drift and maximum
        absolute weight differences between unquantized and quantized models
        across checkpoints (`[Checkpoint Diff]`).

```bash
# Example: Memory-efficient reversible Adam recursion
python -m reversible_data_attribution.run_data_cleansing_eval \
  --mode=all \
  --dataset=mnist \
  --model_type=cnn \
  --optimizer_type=adam \
  --num_train=50000 \
  --num_val=10000 \
  --num_test=10000 \
  --methods=adam_recursive \
  --use_reversible=True \
  --quantization_scale=1000000000000 \
  --output_dir=/tmp/cleansing_reversible
```

#### 3. Adaptive State & HVP Magnitude Clipping (`infl_adam.recursive_update`)

To prevent numerical explosion during backward recursion across deep training
trajectories, `infl_adam.recursive_update` implements **adaptive magnitude
clipping**:

-   **Calibration Phase (First 100 Backward Steps)**: For the initial
    `calib_steps = 100` backward steps (the steps closest to final convergence,
    $t \in [T-100, T-1]$), the algorithm monitors and records the historical
    maximum absolute magnitude of each recurrence term:
    -   Hessian-vector products: `hvp_a`, `hvp_b`
    -   Recurrence accumulator states: `mat_p_new`, `mat_q_new`, `mat_r_new`
-   **Aggressive 5x Thresholding**: For all preceding backward steps ($t < T -
    100$), an adaptive ceiling is enforced at **5x the historical maximum**
    (`clip_factor = 5.0`): $$\text{threshold} = \text{clip\_factor} \times
    \max(\text{calib\_max}, 10^{-4})$$ Any intermediate vector or matrix
    component whose magnitude exceeds this threshold is clamped via
    `torch.clamp(..., -threshold, threshold)`, accompanied by an `Adaptive
    clipping ... at step t` warning in the logs.
-   **Why It's Needed**: In earlier epochs (far from convergence), large
    gradients, non-convex curvature, and division by small variance terms can
    trigger explosive second-order Hessian feedback, driving recursion states to
    `inf` or `NaN`. Aggressive clipping stabilizes the recursion without
    destroying the directional attribution signal.

> [!TIP] **Toggling & Experimenting with Clipping**: We strongly encourage users
> and researchers to experiment with toggling or tuning this clipping behavior
> in `infl_adam.recursive_update`: - **Disable Clipping**: Set
> `clip_factor=None` (or `<= 0`) to run unclipped recursive updates. This is
> useful for evaluating whether unconstrained backward dynamics remain stable
> for well-conditioned models or convex objectives. - **Adjust Headroom**: Vary
> `clip_factor` (e.g., `10.0` or `20.0` for looser bounds, `2.0` for tighter
> stabilization) or adjust the calibration horizon (`calib_steps=50` or `200`)
> to match your model's convergence profile. - **Diagnostics**: If logs show
> frequent clipping warnings (`Adaptive clipping ... at step t: mag > thresh`)
> or if attribution scores appear overly compressed, consider increasing
> `clip_factor` or inspecting gradient norms at the flagged steps.

---

## 3. Parameter Reference (`run_data_cleansing_eval.py`)

### Execution & Sharding Mode

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | `enum` | `'all'` | Execution phase: <br>• `'all'`: Full pipeline (train, compute scores, cleanse & retrain). <br>• `'compute_scores'`: Map phase (compute scores for specified shard and save). <br>• `'cleansing_from_scores'`: Reduce phase (load precomputed scores and evaluate cleansing). |
| `--shard_id` | `int` | `None` | Shard index (0-indexed) for distributed score calculation. |
| `--num_shards` | `int` | `1` | Total number of shards partitioning the dataset. |
| `--save_scores` | `bool` | `True` | Whether to serialize computed scores to disk. |
| `--save_scores_dir`| `string`| `None` | Destination directory for score artifacts (`scores_shard_XXXX.npz`). |
| `--precomputed_scores_dir` | `string` | `None` | Source directory containing precomputed / sharded scores. |
| `--max_workers` | `int` | `1` | Number of worker threads for parallel forward updates and LOO counterfactual evaluation. |

### Attribution & Cleansing Methods

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--methods` | `list` | `adam_recursive,...` | Comma-separated list of methods: <br>`adam_recursive`, `adam_recursive_nodv`, `adam_exact`, `tracin_adam`, `tracin_sgd`, `sgd_all`, `sgd_last`, `icml`, `random`, `ae`, `iso`. |
| `--k_list` | `list` | `0,1,3,6,10,30,60,100` | List of removal counts $k$ for fixed top-$k$ cleansing evaluation. |
| `--auto_cleansing` | `bool` | `True` | Automatically selects and removes all samples whose predicted influence reduces validation loss ($\Delta L < 0$). |
| `--eval_counterfactuals` | `bool` | `False` | Computes ground-truth Leave-One-Out (LOO) counterfactual retraining for all evaluated sample indices. |
| `--retrain_seeds` | `list` | `None` | List of random seeds for randomized counterfactual retraining (evaluates cleansing variance across seeds). |
| `--random_masking` | `float`| `None` | Gradient coordinate Bernoulli($p$) random masking ratio (e.g. `0.05` keeps 5% coordinates) for fast forward Adam approximations (`adam_masked_5%`). |

### Dataset & Corruption

> [!IMPORTANT] **Explicit Sample Counts Required**: It is **necessary to pass in
> `--num_train`, `--num_val`, and `--num_test`**. If omitted, the pipeline
> defaults to small toy sample sizes (1000 train, 200 val, 200 test) rather than
> the complete dataset (e.g. 50,000 train, 10,000 val, 10,000 test for MNIST).
> Always provide these flags explicitly to match your desired dataset split.

| Flag             | Type     | Default           | Description            |
| :--------------- | :------- | :---------------- | :--------------------- |
| `--dataset`      | `enum`   | `'mnist'`         | Dataset: `'mnist'`,    |
:                  :          :                   : `'cifar'`/`'cifar10'`, :
:                  :          :                   : `'cifar100'`,          :
:                  :          :                   : `'fashion_mnist'`,     :
:                  :          :                   : `'imdb'`.              :
| `--data_path`    | `string` | `mnist_local.npz` | Path to `.npz` dataset |
:                  :          :                   : archive.               :
| `--num_train`    | `int`    | `1000`            | **Necessary to pass    |
:                  :          :                   : in.** Number of        :
:                  :          :                   : training samples       :
:                  :          :                   : (e.g., `50000` for     :
:                  :          :                   : full MNIST). Default   :
:                  :          :                   : is a small toy subset  :
:                  :          :                   : (`1000`).              :
| `--num_val`      | `int`    | `200`             | **Necessary to pass    |
:                  :          :                   : in.** Number of        :
:                  :          :                   : validation samples     :
:                  :          :                   : used for query         :
:                  :          :                   : gradient computation   :
:                  :          :                   : (e.g., `10000`).       :
:                  :          :                   : Default is `200`.      :
| `--num_test`     | `int`    | `200`             | **Necessary to pass    |
:                  :          :                   : in.** Number of        :
:                  :          :                   : held-out test samples  :
:                  :          :                   : for evaluation (e.g.,  :
:                  :          :                   : `10000`). Default is   :
:                  :          :                   : `200`.                 :
| `--noise_type`   | `enum`   | `'gaussian'`      | Noise corruption type: |
:                  :          :                   : `'none'`,              :
:                  :          :                   : `'label_flip'`,        :
:                  :          :                   : `'gaussian'`,          :
:                  :          :                   : `'both'`, `'lowpass'`, :
:                  :          :                   : `'highpass'`.          :
| `--noise_rate`   | `float`  | `0.1`             | Fraction of training   |
:                  :          :                   : samples corrupted      :
:                  :          :                   : ($0.0$ to $1.0$).      :
| `--gaussian_std` | `float`  | `0.1`             | Standard deviation for |
:                  :          :                   : additive Gaussian      :
:                  :          :                   : noise.                 :
| `--cutoff_freq`  | `float`  | `0.5`             | Cutoff frequency for   |
:                  :          :                   : lowpass / highpass     :
:                  :          :                   : filtering.             :
| `--noise_seed`   | `int`    | `42`              | Random seed for data   |
:                  :          :                   : corruption, model      :
:                  :          :                   : initialization, and    :
:                  :          :                   : batch ordering.        :

### Model Architecture & Training

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model_type` | `enum` | `'cnn'` | Architecture: `'cnn'`, `'small_dnn'`, `'dnn'`, `'vit'`, `'linear'`, `'resnet'`, `'resnet18'`. |
| `--optimizer_type` | `enum` | `'adam'` | Optimizer: `'adam'` or `'sgd'`. |
| `--learning_rate` | `float` | `0.05` | Learning rate for base model and retraining. |
| `--momentum` | `float` | `0.0` | Momentum for SGD optimizer. |
| `--beta_1` | `float` | `0.9` | Adam $\beta_1$ first moment decay. |
| `--beta_2` | `float` | `0.999` | Adam $\beta_2$ second moment decay. |
| `--eps` | `float` | `1e-8` | Adam numerical stability epsilon. |
| `--num_epochs` | `int` | `10` | Number of training epochs. |
| `--batch_size` | `int` | `64` | Training batch size. |
| `--device` | `string` | `'cpu'` | Hardware device (`'cpu'` or `'cuda'`). |
| `--output_dir` | `string` | `/tmp/data_cleansing_results` | Directory where output CSVs and configs are saved. |

### Memory & Stability Optimizations

| Flag                         | Type   | Default         | Description                     |
| :--------------------------- | :----- | :-------------- | :------------------------------ |
| `--checkpoint_freq`,         | `int`  | `None`          | Step interval for periodic      |
: `--checkpoint_frequency`     :        :                 : checkpointing (e.g. `100`). If  :
:                              :        :                 : `None` and                      :
:                              :        :                 : `--use_reversible=False`, saves :
:                              :        :                 : all step checkpoints. If `None` :
:                              :        :                 : and `--use_reversible=True`,    :
:                              :        :                 : stores only initial (step 0)    :
:                              :        :                 : and final checkpoints.          :
:                              :        :                 : <br>*Note\: Without quantized   :
:                              :        :                 : backward reconstruction         :
:                              :        :                 : (`--use_reversible=True`),      :
:                              :        :                 : periodic checkpointing can      :
:                              :        :                 : amplify floating-point errors   :
:                              :        :                 : in backward recursion methods   :
:                              :        :                 : (e.g.,                          :
:                              :        :                 : `infl_adam.recursive_update`).* :
| `--use_reversible`,          | `bool` | `False`         | Enables reversible Adam with    |
: `--use_reversible_transform` :        :                 : fixed-point quantization to     :
:                              :        :                 : drastically reduce memory usage :
:                              :        :                 : during backward recursion       :
:                              :        :                 : (`adam_recursive`).             :
| `--quantization_scale`       | `int`  | `1000000000000` | Fixed-point quantization scale  |
:                              :        :                 : factor for reversible Adam      :
:                              :        :                 : (default $10^{12}$).            :
| `--ignore_first`             | `bool` | `False`         | Ignores gradient contributions  |
:                              :        :                 : during the first epoch (avoids  :
:                              :        :                 : initial transient gradients;    :
:                              :        :                 : sets `start_step` to number of  :
:                              :        :                 : batches).                       :
| `--start_step`               | `int`  | `0`             | Gradient step cutoff before     |
:                              :        :                 : which influence effects are     :
:                              :        :                 : ignored in recursive updates.   :

#### Function-Level Stability Controls (`infl_adam.recursive_update`)

In addition to the flags above, `infl_adam.recursive_update` exposes parameters
to configure numerical stability and clipping:

-   **`clip_factor`** (default `5.0`): Headroom factor for adaptive state/HVP
    magnitude clipping. Multiplies the maximum magnitude observed during the
    calibration phase (5x historical max). Pass `None` or `<= 0` to disable
    clipping.
-   **`calib_steps`** (default `100`): Number of initial backward steps (nearest
    to convergence) used to calibrate historical maximum magnitudes for `hvp_a`,
    `hvp_b`, `mat_p`, `mat_q`, and `mat_r`.
-   **`eps_hvp`** (default `1e-4`): Curvature damping floor for second-order
    Hessian denominator stabilization.

---

## 4. Output Files & Artifacts

After running an experiment, the output directory contains the following files:

- **`data_cleansing_results.csv`**: Cleansing performance across methods and top-$k$ removal thresholds:
  - Columns: `method`, `k`, `train_loss`, `val_loss`, `test_loss`, `train_acc`, `val_acc`, `test_acc`.
- **`counterfactual_cleansing_summary.csv`**: Automated counterfactual cleansing metrics:
  - Compares `baseline` vs. `cleansed` vs. `oracle` (ground-truth corrupted points removed).
  - Reports `detection_precision`, `detection_recall`, `detection_f1`, and test accuracy/loss deltas.
- **`counterfactual_estimation_summary.csv`**: Attribution accuracy metrics:
  - `roc_auc`: Area under the ROC curve for identifying corrupted samples.
  - `val_loss_corr`, `test_loss_corr`: Correlation between estimated influence and actual leave-one-out loss changes.
  - `mean_val_loss_change_clean` vs. `mean_val_loss_change_corrupted`.
- **`scores.npz` / `scores_shard_XXXX.npz`**: Compressed arrays of computed attribution scores per method.
- **`corrupted_indices.json`**: Ground-truth corrupted index list.
- **`config.json`**: Complete hyperparameter dump.

---

<a id="5-online-data-pruning-framework-next-steps"></a>
## 5. Online Data Pruning Framework (Next Steps)

Whereas standard data cleansing operates **offline** (training a complete base
model, computing scores across all training steps, and retraining from scratch),
the **Online Data Pruning** framework
([`online_pruning.py`](online_pruning.py))
dynamically identifies and removes harmful or corrupted samples **during** the
training process.

### Curriculum Structure ($N = A + B + C$)

The online pruning curriculum is structured into three phases:

1.  **Phase A: Warmup (`warmup_epochs`)**: Trains the base model on all training
    data without pruning to establish foundational representations and a stable
    parameter/optimizer trajectory.
2.  **Phase B: Progressive Pruning (`pruning_epochs`)**: Partitions training
    into moving windows of size `window_size` steps. Within each window, it
    computes data attribution scores across recently executed batches, ranks
    active training points, and progressively masks (prunes) the lowest-scoring
    (most harmful) samples until the cumulative `prune_budget` is reached.
3.  **Phase C: Post-Pruning Stabilization (`post_pruning_epochs`)**: Continues
    training the surviving model solely on the clean, pruned subset to reach
    convergence.
4.  **Retrained Baseline**: Retrains a fresh model from scratch on the final
    pruned dataset (`base.retrain_remove_indices`) to compare the online pruned
    model against a clean retraining baseline.

### Attribution Scoring Options (`score_type`)

`compute_windowed_scores` supports multiple scoring backends over local training
intervals $[m - t, m]$:

-   **`'counterfactual'` (default)**: Forward parameter perturbation tracking
    with Taylor projection onto validation loss gradients.
-   **`'counterfactual_backward'` / `'recursive'`**: Backward recursive
    Hessian-vector product updates via `infl_adam.recursive_update`.
-   **`'tracin'`**: 1-step gradient dot product with validation loss gradients
    across window checkpoints.
-   **`'loss'`**: Loss-based heuristic (higher loss = higher probability of
    removal).
-   **`'random'`**: Random uniform baseline selection.

### How to Run Online Pruning

#### 1. Programmatic Execution (Python / Colab):

```python
from reversible_data_attribution import online_pruning
import torch
from torch import nn

# 1. Configure the online pruning curriculum
config = online_pruning.OnlinePruningConfig(
    warmup_epochs=2,
    pruning_epochs=4,
    post_pruning_epochs=2,
    prune_budget=0.10,            # Prune worst 10% samples
    window_size=10,               # Attribution window size in batches
    score_type='counterfactual',  # 'counterfactual', 'recursive', 'tracin', 'loss', 'random'
    checkpoint_freq=1,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

# 2. Execute curriculum
result = online_pruning.run_online_pruning(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    config=config,
    loss_fn=nn.functional.cross_entropy,
    lr=1e-4,
    corrupted_indices=corrupted_indices,  # Optional: evaluates data_precision / data_recall / data_f1
    seed=42,
)

# 3. Inspect results and metrics
print(f"Total samples pruned: {len(result.pruned_indices)}")
print(f"Online Val Loss: {result.metrics['online_val_loss']:.4f}")
print(f"Retrained Val Loss: {result.metrics['retrain_val_loss']:.4f}")
if 'data_f1' in result.metrics:
  print(f"Corrupted Detection F1: {result.metrics['data_f1']:.4f}")
```

#### 2. Running Unit Tests:

```bash
python -m reversible_data_attribution.online_pruning_test
```

> [!NOTE] **Project Status & Next Steps**: Although the online pruning code and
> curriculum have been executed and verified through unit tests, **systematic
> benchmarking experiments have not yet been run**. Running comprehensive
> empirical sweeps (e.g. evaluating pruning budgets, window horizons, and
> attribution methods against noisy training benchmarks) will be the **next
> step** of this project.
