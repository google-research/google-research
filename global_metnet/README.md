# Global MetNet: Precipitation Nowcasting Model

This directory contains the open-sourced code for our global precipitation
nowcasting model.

For more details on the model architecture, training, and evaluation, please refer to our paper available at: [https://arxiv.org/pdf/2510.13050](https://arxiv.org/pdf/2510.13050)

## Global MetNet Precipitation Nowcasting Dataset

The complete Global MetNet precipitation nowcasting data is made publicly
available for researchers. The dataset offers:

-   **Historical Reforecasts**: High-resolution precipitation nowcasts available from 2020 to present. These reforecasts have a cadence of 3 hours (8 initializations per day).
-   **Real-time Forecasts**: Real-time production forecasts are also available and initialized every 30 minutes.

This data product provides high-resolution (5km spatial, 15 min temporal, up to 12 hr ahead) probabilistic precipitation forecasts, including optimal probability thresholds for conversion to categorical values. The dataset is accessible via a [Google Earth Engine Catalog](https://code.earthengine.google.com/?asset=projects/global-precipitation-nowcast/assets/metnet_nowcast) provided the user agrees to the terms and conditions outlined in this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSeObgf53sXtaZaim08YZYiP_KFTbiIiWYzzs1_LXCeQIKuqlw/viewform).
