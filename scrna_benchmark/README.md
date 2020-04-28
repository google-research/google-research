# Influence of parameters on scRNA-seq dimension reduction methods.

This repository contains the code used for analyzing the influence of parameters
on the performances of dimension reduction (DR) methods.

The methods selected for DR are:

- scran
- seurat
- ZinbWAVE
- DCA
- scVI

## Data used

The data that we used are the Zhengmix4eq, Zhengmix4uneq, and Zhengmix8eq from
[Duo et al.](https://f1000research.com/articles/7-1141). sc\_10x, sc\_10x_5cl,
sc\_celseq2, and sc\_celseq2\_5cl from [Tian et al](https://www.nature.com/articles/s41592-019-0425-8).
And a mixture of the 'FACS' (smart-SEQ2) data from [TabulaMuris](https://www.nature.com/articles/s41586-018-0590-4)
containing all the Brain\_Myeloid, Large\_Intestine, Skin and Spleen cells.

Finally we also created Zhengmix5eq and Zhengmix8uneq with the cell_mixer
software with the following settings:

```bash
Rsript cell_mixer.R \
--data_path=$DATA_PATH \
--name=Zhengmix5eq \
--format=SingleCellExperiment \
--seed=1234 \
--qc_count_mad_lower=3 \
--qc_feature_count_mad_lower=3 \
--qc_mito_mad_upper=3 \
--naive_cytotoxic=1000 \
--regulatory_t=1000 \
--cd4_t_helper=1000 \
--memory_t=1000 \
--naive_t=1000

Rsript cell_mixer.R \
--data_path=$DATA_PATH \
--name=Zhengmix5eq \
--format=SingleCellExperiment \
--seed=1234 \
--qc_count_mad_lower=3 \
--qc_feature_count_mad_lower=3 \
--qc_mito_mad_upper=3 \
--b_cells=500 \
--naive_cytotoxic=250 \
--cd14_monocytes=1000 \
--regulatory_t=1500 \
--cd4_t_helper==500 \
--cd56_nk=250 \
--memory_t=1000 \
--naive_t==1500
```

## Launching the scripts

All the scripts can be run locally (examples in the `run.sh`), however if you
want to do the full benchmark you will nedd a distribution system as it would
take multiple years.

For the R methods, since they don't need GPUs we launched them on google cloud
default machines
using [dsub](https://github.com/DataBiosphere/dsub) which is a similar to qsub.
We provide `generate_dsub_conf.py` which will generate the task files for the
different methods and store both the metrics and the embedding on GCS.
Be warned that launching them will cost you >10k$.
These script will generate one CSV and one Loompy file per configuration.

For the python methods we used another distribution infrastructure. Since
the number of parameter combinations was very high, the launch scripts evaluate
their own grid of parameters and takes ~3 days on a Tesla P100.
These scripts do not save the embedding (by default) as there are simply too
many of them and will write the metrics of multiple runs on a single CSV.
Note that the scripts for scVI and DCA can be interrupted, and will restart
where they left off so that it can be launched on a shared GPU cluster with
preemption.

