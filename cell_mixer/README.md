# Cell Mixer

## Requirements

For this package to work you will need to install the following python3
packages:

```bash
pip3 install pandas anndata absl-py
```

You will also need to install the following R/Bioconductor packages

```R
install.packages("devtools")
install.packages("argparse")
install.packages("Seurat")
install.packages("purrr")
devtools::install_github(repo = "hhoeflin/hdf5r")
devtools::install_github(repo = "mojaveazure/loomR", ref = "develop")

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("SingleCellExperiment")
BiocManager::install("scater")
BiocManager::install("scran")
BiocManager::install("DropletUtils")
```

## Use

### Downloading the data

The current data comes from
[Zheng et al](https://www.nature.com/articles/ncomms14049) and was downloaded
from the
[10X website](https://support.10xgenomics.com/single-cell-gene-expression/datasets).

To download the data, simply run `fetch_data.sh` and specify the path to which
you want to save the data to.

```bash
export DATA_PATH=/tmp/data
bash fetch_data.sh $DATA_PATH
```

Note that the raw data only needs to be fetched once.

### Generating a mixture

The script to generate the mixtures is `cell_mixer.R`, it allows for standard QC
steps:

-   `--qc_counts_mad_lower`: Removing cells with low read count, filtered based
    on the number of MADs under the median read counts.
-   `--qc_feature_count_mas_lower`: Removing cells with few genes expressed,
    filtered based on the number of MADs under the median number of genes
    expressed.
-   `--qc_mito_mad_upper`: Removing cells with high number of mitochondrial
    reads (which is the case for dead cells), based on the number of MADs above
    the median number of mitochondrial RNA counts.

It also allows the select the quantity of cells of various types in the
following table:

Cell type                            | Number of cells | Flag
------------------------------------ | --------------: | ------------------
CD19+ B cells                        | 10085           | --b\_cells
CD8+/CD45RA+ Naive Cytotoxic T Cells | 11953           | --naive\_cytotoxic
CD14+ monocytes                      | 2612            | --cd14\_monocytes
CD4+/CD25+ Regulatory T Cells        | 10263           | --regulatory\_t
CD56+ natural killer cells           | 8385            | --cd56\_nk
CD4+ helper T cells                  | 11213           | --cd4\_t\_helper
CD4+/CD45RO+ Memory T Cells          | 10224           | --memory\_t
CD4+/CD45RA+/CD25- Naive T cells     | 10479           | --naive\_t

The seed for the subsampling is set by default to 1234 in order to reproduce the
data sets from
[Duo et al](https://github.com/csoneson/DuoClustering2018/blob/2b510422c8b799e508b5bbdde93bd8465db2d148/inst/scripts/import_QC_Zhengmix4eq.Rmd#L38),
but it can be changed to generate multiple mixtures with similar cells (for
studying the stability of a result under similar setups).

The cell type identity will be written in the `label` cell attribute.

### Supported formats

This repository can currently generate data in the following formats:

-   [SingleCellExperiment](https://bioconductor.org/packages/release/bioc/html/SingleCellExperiment.html)
-   [Seurat](https://satijalab.org/seurat/)
-   [LoomR/Loompy](https://github.com/mojaveazure/loomR)
-   csv
-   [AnnData](https://github.com/theislab/anndata)

The first four can be done directly with `cell_mixer.R` by specifying the
`--format` flag.

AnnData has to be generate by first generating the data in csv, then by running
the `convert.py` script.

```bash
Rscript cell_mixer.R \
--data_path=$DATA_PATH \
--name=mixture \
--format=csv \
--b_cells=3000 \
--naive_t=3000
python3 converter.py \
--input_csv=mixture \
--format=anndata
```

## Adding new data

In order to add new cell types you can send a Pull Request, the files you will
need to change are:

-   `fetch_data.sh`: to download the data
-   `cell_mixer.R`: add the appropriate flag, read the data, add the label, and
    add it to the mixtures. All the locations to modify have a comment to locate
    them.

## Adding in new formats

The R formats have to be added in the `cell_mixer.R` script, internally it uses
SingleCellExperiment which is the most commonly used format.

The python formats have to be added in `converter.py`.

If you want more formats to be supported please open an issue or send a pull
request.
