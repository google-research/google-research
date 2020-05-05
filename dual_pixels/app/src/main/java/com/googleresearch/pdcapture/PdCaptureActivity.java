package com.googleresearch.pdcapture;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Gravity;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.ShortBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.TimeZone;

/** An activity to capture and save Dual-Pixel data. */
public class PdCaptureActivity extends Activity {
  private static final String TAG = "PdCaptureApp";
  // These are the dimensions of the dual-pixel data. The sensor has 2016x1512 2x2 Bayer quads
  // (https://en.wikipedia.org/wiki/Bayer_filter). Each 2x2 quad consists of one pixel that records
  // red light, one that records blue light and two pixels that record green light. When demoasiced,
  // these pixels can produce a 4032x3024 RGB image. The green pixels are also split in half and
  // binned such that the four left and right green half-pixels in adjacent Bayer quads are averaged
  // together. This results in dual-pixel data that is one quarter the size of the RGB image
  // vertically and one half the size horizontally. The data entirely comes from the green channel.
  private static final int DP_WIDTH = 2016;
  private static final int DP_HEIGHT = 756;
  private static final int DP_CHANNELS = 2;
  /**
   * Unexposed ImageFormat corresponding to dual-pixel data. This is a hidden member of the enum
   * {@link android.graphics.ImageFormat}.
   */
  private static final int IMAGE_FORMAT_DP = 0x1002;

  private static final int IMAGE_READER_DP_WIDTH = DP_WIDTH * DP_CHANNELS;
  private static final int IMAGE_READER_DP_HEIGHT = DP_HEIGHT;

  /** The white level of dual-pixel data as recorded by the camera. */
  private static final int DP_WHITE_LEVEL = 1023;

  /** The white level we will save dual-pixel data with. */
  private static final int TARGET_WHITE_LEVEL = 65535;

  // User interface.
  private Button captureDpButton;
  private TextureView viewfinderTextureView;

  // Camera controller objects.
  private CameraDevice cameraDevice;
  private CameraCaptureSession cameraCaptureSession;
  private HandlerThread cameraThread;
  private Handler cameraHandler;
  private ImageReader dpImageReader;
  private ImageReader rawImageReader;
  private Size previewDimensions;
  private Size raw10Dimensions;
  private File outputDirectory;

  // Permission
  private static int CAMERA_PERMISSION_REQUEST_CODE = 0;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    viewfinderTextureView = findViewById(R.id.texture);
    viewfinderTextureView.setSurfaceTextureListener(surfaceTextureListener);
    captureDpButton = findViewById(R.id.captureDpButton);
    // We don't use lambda expressions to avoid requiring Java8.
    captureDpButton.setOnClickListener(
        new View.OnClickListener() {
          public void onClick(View v) {
            captureDpData();
          }
        });
    outputDirectory = getApplicationContext().getExternalFilesDir("");
  }

  @Override
  public void onPause() {
    cameraThread.quitSafely();
    try {
      cameraThread.join();
      cameraThread = null;
      cameraHandler = null;
    } catch (InterruptedException e) {
      Log.e(TAG, "Unable to stop camera thread: ", e);
    }
    closeCamera();
    super.onPause();
  }

  @Override
  public void onResume() {
    super.onResume();
    cameraThread = new HandlerThread("CameraThread");
    cameraThread.start();
    cameraHandler = new Handler(cameraThread.getLooper());
    if (viewfinderTextureView.isAvailable()) {
      openCamera();
    } else {
      viewfinderTextureView.setSurfaceTextureListener(surfaceTextureListener);
    }
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
      if (grantResults.length != 1 || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
        AlertDialog alertDialog = new AlertDialog.Builder(PdCaptureActivity.this).create();
        alertDialog.setTitle("Camera Permission");
        alertDialog.setMessage("Camera Permission is not granted");
        alertDialog.setButton(
            AlertDialog.BUTTON_NEUTRAL,
            "OK",
            new DialogInterface.OnClickListener() {
              public void onClick(DialogInterface dialog, int which) {
                dialog.dismiss();
              }
            });
        alertDialog.show();
      }
    } else {
      super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }
  }

  /** Creates the camera device and the image reader. */
  private void openCamera() {
    CameraManager manager = (CameraManager) getSystemService(CAMERA_SERVICE);
    try {
      String cameraId = getFirstRearCameraId(manager);
      CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
      StreamConfigurationMap streamConfigMap =
          characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
      previewDimensions = getLargestByArea(streamConfigMap.getOutputSizes(SurfaceTexture.class));
      raw10Dimensions = getLargestByArea(streamConfigMap.getOutputSizes(ImageFormat.RAW10));
      Log.d(TAG, "Preview dimensions: " + previewDimensions);
      if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
        requestPermissions(
            new String[] {Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
        return;
      }
      manager.openCamera(cameraId, stateCallback, null);
      createImageReader();
    } catch (CameraAccessException e) {
      Log.e(TAG, "Error opening camera: ", e);
    } catch (NoSuchElementException e) {
      Log.e(TAG, "Error opening camera: ", e);
    }
  }

  /**
   * Gets the ID of the first rear camera.
   *
   * @throws NoSuchElementException if no rear cameras are found.
   */
  private String getFirstRearCameraId(CameraManager cameraManager)
      throws CameraAccessException, NoSuchElementException {
    String[] cameraIdList = cameraManager.getCameraIdList();
    for (String id : cameraIdList) {
      if (cameraManager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING)
          == CameraCharacteristics.LENS_FACING_BACK) {
        return id;
      }
    }

    throw new NoSuchElementException("Couldn't find a rear facing camera.");
  }

  /**
   * Starts the camera's preview. Configures both a viewfinder stream and a dual-pixel stream, but
   * only requests captures from the viewfinder stream.
   */
  private void startPreview() {
    try {
      SurfaceTexture texture = viewfinderTextureView.getSurfaceTexture();
      texture.setDefaultBufferSize(previewDimensions.getWidth(), previewDimensions.getHeight());
      // The image stream is in landscape orientation, but we force the texture to be in portrait
      // orientation. Therefore, we reverse height and width here.
      adjustAspectRatio(previewDimensions.getHeight(), previewDimensions.getWidth());
      final Surface viewfinderSurface = new Surface(texture);
      List<Surface> outputSurfaces = new ArrayList<Surface>(3);
      outputSurfaces.add(dpImageReader.getSurface());
      outputSurfaces.add(rawImageReader.getSurface());
      outputSurfaces.add(viewfinderSurface);
      cameraDevice.createCaptureSession(
          outputSurfaces,
          new CameraCaptureSession.StateCallback() {
            @Override
            public void onConfigured(CameraCaptureSession session) {
              Log.d(TAG, "Entering CameraCaptureSession.StateCallback.onConfigured()");
              if (null == cameraDevice) {
                Log.e(TAG, "cameraDevice is null when attempting to start repeating request.");
                return;
              }
              cameraCaptureSession = session;
              try {
                CaptureRequest.Builder captureRequestBuilder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                captureRequestBuilder.addTarget(viewfinderSurface);
                captureRequestBuilder.set(
                    CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
                cameraCaptureSession.setRepeatingRequest(
                    captureRequestBuilder.build(), null, cameraHandler);
              } catch (CameraAccessException e) {
                Log.e(TAG, "Error starting repeating request: ", e);
              }
            }

            @Override
            public void onConfigureFailed(CameraCaptureSession session) {
              Log.e(TAG, "Failure while opening camera.");
            }
          },
          null);
    } catch (CameraAccessException e) {
      Log.e(TAG, "Error starting preview: ", e);
    }
  }

  private void closeCamera() {
    if (null != cameraDevice) {
      cameraDevice.close();
      cameraDevice = null;
    }
    if (null != dpImageReader) {
      dpImageReader.close();
      dpImageReader = null;
    }
    if (null != rawImageReader) {
      rawImageReader.close();
      rawImageReader = null;
    }
  }

  /** Inserts a request to capture dual-pixel data into the queue of capture requests. */
  private void captureDpData() {
    try {
      final CaptureRequest.Builder captureBuilder =
          cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);
      captureBuilder.addTarget(dpImageReader.getSurface());
      captureBuilder.addTarget(new Surface(viewfinderTextureView.getSurfaceTexture()));
      captureBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
      cameraCaptureSession.capture(captureBuilder.build(), null, cameraHandler);
    } catch (CameraAccessException e) {
      Log.e(TAG, "CameraAccessException: ", e);
    }
  }

  /** Configures an ImageReader for dual-pixel data. */
  private void createImageReader() {
    dpImageReader =
        ImageReader.newInstance(
            IMAGE_READER_DP_WIDTH, IMAGE_READER_DP_HEIGHT, IMAGE_FORMAT_DP, /*maxImages=*/ 10);
    dpImageReader.setOnImageAvailableListener(
        new ImageReader.OnImageAvailableListener() {
          @Override
          public void onImageAvailable(ImageReader reader) {
            Image image = null;
            try {
              image = reader.acquireLatestImage();
              Log.d(TAG, "Acquired image with timestamp: " + image.getTimestamp());
              ShortBuffer buffer = image.getPlanes()[0].getBuffer().asShortBuffer();
              short[] interleavedDpData = new short[buffer.capacity()];
              buffer.get(interleavedDpData);

              DpData dpData =
                  deinterleaveAndScaleWhiteLevel(
                      interleavedDpData, DP_WHITE_LEVEL, TARGET_WHITE_LEVEL);

              // Saves images with filenames based on the current time.
              String rootName = getCurrentTimeStr();
              Toast toast =
                  Toast.makeText(PdCaptureActivity.this, "Saved:" + rootName, Toast.LENGTH_SHORT);
              toast.setGravity(Gravity.TOP, 0, 0);
              toast.show();
              File outputLeft = new File(outputDirectory, rootName + "_left.pgm");
              savePgm16(outputLeft.getAbsolutePath(), dpData.leftView, DP_WIDTH, DP_HEIGHT);
              File outputRight = new File(outputDirectory, rootName + "_right.pgm");
              savePgm16(outputRight.getAbsolutePath(), dpData.rightView, DP_WIDTH, DP_HEIGHT);
            } finally {
              if (image != null) {
                image.close();
              }
            }
          }
        },
        cameraHandler);
    // Some devices require a RAW10 image reader for DP capture to work.
    rawImageReader =
        ImageReader.newInstance(
            raw10Dimensions.getWidth(),
            raw10Dimensions.getHeight(),
            ImageFormat.RAW10,
            /*maxImages=*/ 10);
  }

  /** Holds data corresponding to the left and right dual-pixel views. */
  private class DpData {
    // This data spans the range [0, 65535]. The fact that short has a range of [-32768, 32767]
    // doesn't matter since after scaling, we only deal with the byte representation of the data.
    public short[] leftView;
    public short[] rightView;
  }

  /** Deinterleaves dual-pixel data into two buffers representing the left and right views. */
  private DpData deinterleaveAndScaleWhiteLevel(
      short[] interleavedDpData, int inputWhiteLevel, int outputWhiteLevel) {
    DpData dpData = new DpData();
    final int viewSize = interleavedDpData.length / 2;
    dpData.leftView = new short[viewSize];
    dpData.rightView = new short[viewSize];
    final float multiplier = (float) outputWhiteLevel / inputWhiteLevel;
    for (int i = 0; i < viewSize; ++i) {
      dpData.leftView[i] = (short) (interleavedDpData[2 * i] * multiplier);
      dpData.rightView[i] = (short) (interleavedDpData[2 * i + 1] * multiplier);
    }
    return dpData;
  }

  /**
   * Converts a short array to a byte array by turning every short into its big-endian two byte
   * representation.
   */
  private byte[] shortArrayToByteArray(short[] array) {
    byte[] bytes = new byte[array.length * 2];
    for (int i = 0; i < array.length; ++i) {
      bytes[2 * i] = (byte) ((array[i] >> 8) & 0xff);
      bytes[2 * i + 1] = (byte) (array[i] & 0xff);
    }
    return bytes;
  }

  /** Saves a 16 bit image as a PGM file. */
  private void savePgm16(String filename, short[] imageData, int width, int height) {
    String header = String.format("P5 %d %d %d ", width, height, TARGET_WHITE_LEVEL);
    OutputStream output = null;
    try {
      output = new FileOutputStream(filename);
      if (null != output) {
        output.write(header.getBytes());
        output.write(shortArrayToByteArray(imageData));
        output.close();
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  /** Returns current date time as a string. */
  private String getCurrentTimeStr() {
    SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyyMMdd_HHmmss_SSS");
    simpleDateFormat.setTimeZone(TimeZone.getDefault());
    return simpleDateFormat.format(System.currentTimeMillis());
  }

  /**
   * {@link TextureView.SurfaceTextureListener} handles several lifecycle events on a {@link
   * TextureView}.
   */
  private final TextureView.SurfaceTextureListener surfaceTextureListener =
      new TextureView.SurfaceTextureListener() {

        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture texture, int width, int height) {
          Log.d(TAG, "Surface texture available.");
          openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture texture, int width, int height) {}

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture texture) {
          return true;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture texture) {}
      };

  private final CameraDevice.StateCallback stateCallback =
      new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
          Log.d(TAG, "Camera has been opened.");
          cameraDevice = camera;
          startPreview();
        }

        @Override
        public void onDisconnected(CameraDevice camera) {
          Log.e(TAG, "Camera has been disconnected.");
          cameraDevice.close();
        }

        @Override
        public void onError(CameraDevice camera, int error) {
          cameraDevice.close();
          cameraDevice = null;
        }
      };

  /** Get the largest element of sizes sorted by area. */
  private Size getLargestByArea(Size[] sizes) {
    assert sizes.length != 0;
    int largestArea = -1;
    Size largestSize = sizes[0];
    for (Size size : sizes) {
      final int area = size.getWidth() * size.getHeight();
      if (area > largestArea) {
        largestArea = area;
        largestSize = size;
      }
    }
    return largestSize;
  }

  /** Adjusts the aspect ratio of viewfinderTextureView to match that of the viewfinder stream. */
  private void adjustAspectRatio(int desiredWidth, int desiredHeight) {
    int width = viewfinderTextureView.getWidth();
    int height = viewfinderTextureView.getHeight();
    int newWidth = width;
    int newHeight = height;
    if (width * desiredHeight > height * desiredWidth) {
      newWidth = newHeight * desiredWidth / desiredHeight;
    } else {
      newHeight = newWidth * desiredHeight / desiredWidth;
    }
    Matrix transform = new Matrix();
    viewfinderTextureView.getTransform(transform);
    transform.setScale((float) newWidth / width, (float) newHeight / height);
    viewfinderTextureView.setTransform(transform);
  }
}
