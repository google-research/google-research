# High Performance Monte Carlo Simulation of Ising Model on TPU Clusters

This is the implementation accompanying the paper (accepted by ACM/IEEE SC2019):
["High Performance Monte Carlo Simulation of Ising Model on TPU Clusters"](
https://arxiv.org/abs/1903.11714). This is the exact implementation used in the
work.

The results reported in the paper are based on `TensorFlow r1.13` release and
the performance numbers are measured with profiling tools on the traces of
operations directly.  The same profiling tools are also available in Google
Cloud and can be setup to do the same measurements.

On the other hand, this notebook also offers a more direct way to estimate the
performance without the profiling tools by measuring the wall-clock step time.
Although the direct wall-clock time measurements will include other overhead
cost (Just-In-Time `XLA` compilation, one-time network latency between notebook
kernel and the `TPU` cluster etc.), these are amortizable over large number of
steps and an estimation of the overhead can be obtained so the true step time
can be inferred.

## Simple Example Usage

Following the instructions below, you can quickly run the same program used in
the paper, at a smaller scale with an older version of `TPU (TPUv2)`. You will
not be able to reproduce the exact results but it provides good insights:

1. Go to https://colab.research.google.com

2. In the pop-up window, select tab `GITHUB` and enter
https://github.com/google-research/google-research, then search for
`ising_mcmc_tpu.ipynb` and click to load the notebook into colaboratory.

3. In menu Edit -> Notebook settings -> Hardware accelerator, select `TPU`.

4. Then click CONNECT, and you will be able to simulate Ising model.

As pointed out, the free `TPU` available from http://colab.research.google.com
is `TPUv2` and there are at most 8 `TPU` cores available. To more closely repeat
our experiments in the paper, `TPUv3` is needed and you would also need a
larger number of cores. Follow the instructions in the next section to set up
a Google Cloud project for this.

Note that as of 2019/08/27, the `TensorFlow version` used in
http://colab.research.google.com is `r1.14`, which has a bug that breaks this
program. A fix has been implemented for `r1.15` and once it is released, the
issue should be resolved. For full setup (see below), users can choose to use
`r1.13` to avoid such a problem.

## Full Setup on Google Cloud

The following are the instructions to set up to run at full scale.

1. Create a Google Cloud project. Detailed steps for doing this can be found
[here](https://cloud.google.com/dataproc/docs/guides/setup-project).

2. Set up a `TPU` instance and and the corresponding virtual machine (`VM`)
instance in the Cloud project. You can follow the steps [here](
https://cloud.google.com/tpu/docs/quickstart).  Note that the experiments
reported are done using `TensorFlow r1.13` release (and there is a bug in
`r1.14` that would break the program). Please select the appropriate software
version when creating the instance (`r1.13` or `r1.15`, if available,
recommended).

3. To run the larger scale experiments described in the paper using `TPUv3`,
you may need to request beta quota. Use this page to file a request:
https://cloud.google.com/contact/.

4. On your workstation, install `ctpu` (see [here](
https://github.com/tensorflow/tpu/tree/master/tools/ctpu) ) and `gcloud`
(https://cloud.google.com/sdk/install) as they are useful for connecting to the
cloud instances.

5. You can ssh to your `VM` using Cloud Console's `SSH` drop-down manual. It
also provides a way to upload files to your `VM`. You can use this functionality
to upload this notebook to the `VM`.

6. On your `VM`, check that the environment variable `TPU_NAME` is set to the
`TPU` cluster instance name.

7. On your `VM`, start a `Jupyter` notebook kernel:

   ```shell
   jupyter notebook --port=8888
   ```

8. Forward `port 8888` of the `VM` to your local workstation. You can do so by:

    ```shell
    gcloud compute ssh $vm_instance_name$ --zone $instance_zone$ --project=$project_name$ -- -L 8888:localhost:8888
    ```
9. On your local workstation, open a browser and point it to
`https://localhost:8888` (you might also need to include the security token
that was printed out when you started the Jupyter notebook kernel). You should
be able to run the experiments through it.

Citing
------
```none
@ARTICLE{yang2019isingtpu,
  author = {Yang, Kun and Chen, Yi-Fan and Roumpos, Georgios and Colby, Chris and Anderson, John},
  title = {High Performance Monte Carlo Simulation of Ising Model on TPU Clusters},
  journal = {arXiv preprint arXiv: 1903.11714},
  year = {2019}
}
