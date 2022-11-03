package com.googleresearch.bustle;

import com.google.common.collect.ImmutableList;
import com.googleresearch.bustle.propertysignatures.ComputeSignature;
import com.googleresearch.bustle.propertysignatures.PropertySummary;
import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.Value;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/**
 * Checks that SavedModelWrapper can load up a saved model and do inference. Also serves as an
 * example of how to call SavedModelWrapper that we can use when modifying the synthesizer to use
 * inference. This really should be a test, but writing it like this was easier.
 */
@Command(name = "CheckSavedModelWrapper", mixinStandardHelpOptions = true)
final class CheckSavedModelWrapper implements Runnable {

  @SuppressWarnings("FieldCanBeFinal") // picocli modifies value when flag is set
  @Option(
      names = {"--saved_model_file"},
      description = "Directory containing the saved model.")
  private static String savedModelFile = "/tmp/saved_model_dir/";

  @Override
  public void run() {
    System.out.println(savedModelFile);
    SavedModelWrapper savedModelWrapper;
    try {
      savedModelWrapper = new SavedModelWrapper(savedModelFile);
    } catch (IOException e) {
      System.err.println(e);
      return;
    }

    System.out.println("Input names and shapes: ");
    System.out.println(savedModelWrapper.getInputNames());
    System.out.println(savedModelWrapper.getInputShapes());
    System.out.println("Output names and shapes: ");
    System.out.println(savedModelWrapper.getOutputNames());
    System.out.println(savedModelWrapper.getOutputShapes());

    ImmutableList<InputValue> inputs =
        ImmutableList.of(new InputValue(Arrays.asList("butter", "bunter", "x"), "inputs"));
    Value intermediateValue =
        new InputValue(Arrays.asList("butterfl", "bunterfl", "xxy"), "inputs");
    Value outputValue = new InputValue(Arrays.asList("butterfly", "butterfly", "xx"), "outputs");

    List<PropertySummary> exampleSignature =
        ComputeSignature.computeExampleSignature(inputs, outputValue);
    List<PropertySummary> valueSignature =
        ComputeSignature.computeValueSignature(intermediateValue, outputValue);

    System.out.println("exampleSignature: " + exampleSignature);
    System.out.println("exampleSignature size: " + exampleSignature.size());
    System.out.println("valueSignature: " + valueSignature);
    System.out.println("valueSignature size: " + valueSignature.size());

    // Feed in the first matrix to get the first output
    float[][] outputOne =
        savedModelWrapper.doInference(exampleSignature, ImmutableList.of(valueSignature));

    // reverse signatures to check that we get different outputs
    Collections.reverse(valueSignature);
    float[][] outputTwo =
        savedModelWrapper.doInference(exampleSignature, ImmutableList.of(valueSignature));

    // Check that the outputs are different, given the different inputs
    System.out.println(outputOne[0][0]);
    System.out.println(outputTwo[0][0]);
  }

  public static void main(String[] args) {
    int exitCode = new CommandLine(new CheckSavedModelWrapper()).execute(args);
    System.exit(exitCode);
  }

  private CheckSavedModelWrapper() {}
}
