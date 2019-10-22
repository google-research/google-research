# Learning Single Camera Depth Estimation using Dual-Pixels

Reference code for the paper [Learning Single Camera Depth Estimation using Dual-Pixels](https://arxiv.org/abs/1904.05822).
Rahul Garg, Neal Wadhwa, Sameer Ansari & Jonathan T. Barron, ICCV 2019. If you use this code or our dataset, please cite our paper:
```
@article{GargDualPixelsICCV2019,
  author    = {Rahul Garg and Neal Wadhwa and Sameer Ansari and Jonathan T. Barron},
  title     = {Learning Single Camera Depth Estimation using Dual-Pixels},
  journal   = {ICCV},
  year      = {2019},
}
```


## Dataset

Coming soon!

## Android App to Capture Dual-pixel Data

The app has been tested on the Google Pixel 3 and Pixel 4.

### Installation instructions:

1. Download [Android Studio](https://developer.android.com/studio). When you install it, make sure to also install the Android SDK API 29.
2. Click "Open an existing Android Studio project". Select the "dual_pixels" directory.
3. There will be a popup with title "Gradle Sync" complaining about a missing file called gradle-wrapper.properties. Click ok to recreate the Gradle wrapper.
4. Plug in your Pixel smartphone. You'll need to enable USB debugging. See
https://developer.android.com/studio/debug/dev-options for further instructions.
5. Go to the "Run" menu at the top and click "Run 'app'" to compile and install the app.

Dual-pixel data will be saved in the directory:
```
/sdcard/Android/data/com.google.reseach.pdcapture/files
```

### Information about saved data

The images are captured with a white level of 1023 (10 bits), but the app
linearly scales them to have white level 65535. That is, the pixels are
initially in the range \[0, 1023\], but are scaled to the range \[0, 65535\].
These images are in linear space. That is, they have not been gamma-encoded. The
black level is typically around 64 before scaling and 1024 after scaling. The
exact black level can be obtained via [SENSOR_DYNAMIC_BLACK_LEVEL](https://developer.android.com/reference/android/hardware/camera2/CaptureResult.html#SENSOR_DYNAMIC_BLACK_LEVEL) in the CaptureResult.

This app only saves the dual-pixel images. It is also possible to save files in
ImageFormat.RAW10 and metadata captured at the same time. Other image stream
combinations may not work.

This app will not work on non-Google phones and will not work on any phone
released by Google prior to the Pixel 3. It should work with the 3a and 4
running Android 10, but is not guaranteed to work on any Pixel phones after that
point.


## Related Work
Wadhwa et al., [Synthetic Depth-of-Field with a Single-Camera Mobile Phone](https://arxiv.org/abs/1806.04171),
SIGGRAPH 2018


