package com.googleresearch.bustle;

import com.google.protobuf.InvalidProtocolBufferException;
import com.googleresearch.bustle.exception.SynthesisError;
import com.googleresearch.bustle.propertysignatures.PropertySummary;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.tensorflow.framework.TensorShapeProto.Dim;

/** Loads up a saved tensorflow model that we trained in python. */
public class SavedModelWrapper {
  private final Session sess;
  private final List<String> inputNames;
  private final List<List<Long>> inputShapes;
  private final List<String> outputNames;
  private final List<List<Long>> outputShapes;

  // NOTE! This class assumes the input is a 2-d float tensor.
  public SavedModelWrapper(String savedModelDirectory) throws InvalidProtocolBufferException {
    SavedModelBundle savedModelBundle = SavedModelBundle.load(savedModelDirectory, "serve");
    Session sess = savedModelBundle.session();

    SignatureDef sig;
    try {
      sig =
          MetaGraphDef.parseFrom(savedModelBundle.metaGraphDef())
              .getSignatureDefOrThrow("serving_default");
    } catch (InvalidProtocolBufferException e) {
      System.out.println("Failed to load saved model from dir: " + savedModelDirectory);
      throw e;
    }
    this.sess = sess;
    this.inputNames = new ArrayList<>();
    this.inputShapes = new ArrayList<>();
    this.outputNames = new ArrayList<>();
    this.outputShapes = new ArrayList<>();
    getNamesAndShapesFromMap(sig.getInputsMap(), this.inputNames, this.inputShapes);
    getNamesAndShapesFromMap(sig.getOutputsMap(), this.outputNames, this.outputShapes);

    int inputDims = inputShapes.get(0).size();
    if (inputDims != 2) {
      throw new SynthesisError(
          "Input Shape from SavedModel has wrong number of dimensions: " + inputDims);
    }
  }

  private Tensor<Float> inputTensorFromSignatures(
      List<PropertySummary> inputOutputSignature,
      List<List<PropertySummary>> intermediateSignatures) {
    int exampleSignatureSize = inputOutputSignature.size();
    int intermediateSignatureSize = intermediateSignatures.get(0).size();
    int inputSize = inputShapes.get(0).get(1).intValue();
    int batchSize = intermediateSignatures.size();
    if (exampleSignatureSize + intermediateSignatureSize != inputSize) {
      throw new SynthesisError(
          "example sig size "
              + exampleSignatureSize
              + " + intermediate sig size "
              + intermediateSignatureSize
              + " should equal input size "
              + inputSize);
    }
    long[] shape = new long[] {batchSize, inputSize};
    FloatBuffer buf = FloatBuffer.allocate(batchSize * inputSize);

    float[] inputOutputSignatureAsInts = new float[exampleSignatureSize];
    for (int i = 0; i < exampleSignatureSize; i++) {
      inputOutputSignatureAsInts[i] = inputOutputSignature.get(i).asInt();
    }

    for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
      List<PropertySummary> intermediateSummary = intermediateSignatures.get(batchIndex);
      // Fill in the first half with the input/output signature.
      buf.put(inputOutputSignatureAsInts);
      // Fill in the 2nd half with intermediate value signature.
      for (int i = 0; i < intermediateSignatureSize; i++) {
        buf.put(intermediateSummary.get(i).asInt());
      }
    }
    buf.flip();
    return Tensor.create(shape, buf);
  }

  private static void getNamesAndShapesFromMap(
      Map<String, TensorInfo> map, List<String> names, List<List<Long>> shapes) {
    for (Map.Entry<String, TensorInfo> entry : map.entrySet()) {
      names.add(entry.getValue().getName());
      List<Long> dimList = new ArrayList<>();
      for (Dim dim : entry.getValue().getTensorShape().getDimList()) {
        dimList.add(dim.getSize());
      }
      shapes.add(dimList);
    }
  }

  public List<List<Long>> getInputShapes() {
    return inputShapes;
  }

  public List<String> getInputNames() {
    return inputNames;
  }

  public List<String> getOutputNames() {
    return outputNames;
  }

  public List<List<Long>> getOutputShapes() {
    return outputShapes;
  }

  public float[][] doInference(
      List<PropertySummary> inputOutputSignature,
      List<List<PropertySummary>> intermediateSignatures) {
    int batchSize = intermediateSignatures.size();
    // This code assumes that there is a single 2-D output!
    int outputDim = (int) (long) outputShapes.get(0).get(1);
    Tensor<Float> inputTensor =
        inputTensorFromSignatures(inputOutputSignature, intermediateSignatures);
    Tensor<?> outputTensor =
        sess.runner().feed(inputNames.get(0), inputTensor).fetch(outputNames.get(0)).run().get(0);
    float[][] outputMatrix = new float[batchSize][outputDim];
    outputTensor.copyTo(outputMatrix);
    return outputMatrix;
  }
}
