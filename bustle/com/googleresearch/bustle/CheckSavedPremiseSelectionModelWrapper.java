package com.googleresearch.bustle;

import com.google.common.collect.ImmutableList;
import com.googleresearch.bustle.propertysignatures.ComputeSignature;
import com.googleresearch.bustle.propertysignatures.PropertySummary;
import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.Value;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/** Checks that we can load up a saved model and do inference for *premise selection*. */
@Command(name = "CheckSavedPremiseSelectionModelWrapper", mixinStandardHelpOptions = true)
final class CheckSavedPremiseSelectionModelWrapper implements Runnable {

  @SuppressWarnings("FieldCanBeFinal") // picocli modifies value when flag is set
  @Option(
      names = {"--saved_premise_selection_model_file"},
      description = "Model location.")
  private static String savedModelFile = "/tmp/saved_premise_selection_model_dir/";

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

    // Print out the outputs
    for (int i = 0; i < savedModelWrapper.getOutputShapes().get(0).get(1); i++) {
      System.out.println(outputOne[0][i]);
    }
  }

  public static void main(String[] args) {
    int exitCode = new CommandLine(new CheckSavedPremiseSelectionModelWrapper()).execute(args);
    System.exit(exitCode);
  }

  private CheckSavedPremiseSelectionModelWrapper() {}
}
