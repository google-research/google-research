# scHPTM benchmark

This repository contains the code for benchmarking the quality of the
representation of single-cell histone post translational modifications (scHPTM).

We evaluate the effect of the following factors:

-   Matrix construction.
-   Quality control (based on coverage).
-   Feature selection effect (using HVG or coverage).
-   Role of number of cells per experiment (by downsampling).
-   Role of coverage per cell (by downsampling cells based on coverage).
-   Role of normalization used (RPM or TF-IDF).
-   Role of dimension reduction algorithm (using 7 standard single-cell
    epigenetics pipelines).

The evaluation relies on having a robust co-assay (either transcriptomic or
surface proteins), and measuring how well the scHPTM representation conserves
the local geometry that can be observed in the reference co-assay.
